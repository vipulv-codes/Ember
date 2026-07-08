import copy

import torch

from lora import apply_lora, lora_state_dict, merge_lora
from model import Ember, EmberConfig

TINY = EmberConfig(vocab_size=101, ctx_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)
R, ALPHA = 4, 8.0


def make_pair():
    torch.manual_seed(0)
    base = Ember(TINY).eval()
    adapted = copy.deepcopy(base)
    n_trainable = apply_lora(adapted, r=R, alpha=ALPHA)
    return base, adapted, n_trainable


def test_trainable_count_is_2rd_per_matrix():
    _, adapted, n_trainable = make_pair()
    # wq + wv per layer, each square d x d -> 2*r*d params per matrix
    expected = TINY.n_layers * 2 * (2 * R * TINY.d_model)
    assert n_trainable == expected
    assert all(
        p.requires_grad == (".A" in n or ".B" in n) for n, p in adapted.named_parameters()
    )


def test_identity_at_init():
    """B=0 means the adapted model IS the base model before training."""
    base, adapted, _ = make_pair()
    idx = torch.randint(0, TINY.vocab_size, (2, 32))
    with torch.no_grad():
        lb, _ = base(idx)
        la, _ = adapted(idx)
    assert torch.equal(lb, la)


def test_training_moves_adapters_not_base():
    base, adapted, _ = make_pair()
    idx = torch.randint(0, TINY.vocab_size, (2, 32))
    opt = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        _, loss = adapted(idx, targets=idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # outputs changed...
    with torch.no_grad():
        lb, _ = base(idx)
        la, _ = adapted(idx)
    assert not torch.equal(lb, la)
    # ...but every frozen weight is bit-identical to the base model's
    bsd = base.state_dict()
    for k, v in adapted.state_dict().items():
        plain = k.replace(".base.weight", ".weight")
        if ".A" not in k and ".B" not in k:
            assert torch.equal(v, bsd[plain]), f"frozen weight moved: {k}"


def test_merge_preserves_outputs_and_removes_overhead():
    _, adapted, _ = make_pair()
    idx = torch.randint(0, TINY.vocab_size, (2, 32))
    # train a little so the adapters are non-trivial
    opt = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=1e-2)
    _, loss = adapted(idx, targets=idx)
    loss.backward()
    opt.step()
    with torch.no_grad():
        before, _ = adapted(idx)
    merged = merge_lora(adapted)
    with torch.no_grad():
        after, _ = merged(idx)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    assert not any(".A" in k or ".B" in k for k in merged.state_dict())


def test_lora_state_dict_is_small():
    _, adapted, n_trainable = make_pair()
    sd = lora_state_dict(adapted)
    assert sum(v.numel() for v in sd.values()) == n_trainable
