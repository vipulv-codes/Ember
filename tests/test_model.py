import pytest
import torch

from model import Ember, EmberConfig, preset_config, PRESETS

# Small config so CPU tests run in seconds.
TINY = EmberConfig(vocab_size=101, ctx_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)


def test_forward_shapes():
    torch.manual_seed(0)
    model = Ember(TINY)
    idx = torch.randint(0, TINY.vocab_size, (3, 32))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (3, 32, TINY.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


@pytest.mark.parametrize("name", list(PRESETS))
def test_param_count_matches_hand_count(name):
    cfg = preset_config(name)
    model = Ember(cfg)
    # Weight tying: lm_head shares the embedding tensor, count unique params.
    unique = sum(
        t.numel() for t in {p.data_ptr(): p for p in model.parameters()}.values()
    )
    assert unique == cfg.n_params(), f"{name}: {unique} != hand count {cfg.n_params()}"


def test_hero_is_about_30m():
    assert abs(preset_config("ember-30m").n_params() - 30e6) / 30e6 < 0.05


def test_naive_equals_sdpa():
    torch.manual_seed(0)
    model = Ember(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 48))
    with torch.no_grad():
        model.cfg.attn_impl = "naive"
        logits_naive, _ = model(idx)
        model.cfg.attn_impl = "sdpa"
        logits_sdpa, _ = model(idx)
    torch.testing.assert_close(logits_naive, logits_sdpa, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("impl", ["naive", "sdpa"])
def test_causal_mask_no_leak(impl):
    """Perturbing a future token must not change logits at earlier positions."""
    torch.manual_seed(0)
    cfg = EmberConfig(**{**TINY.__dict__, "attn_impl": impl})
    model = Ember(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 32))
    idx2 = idx.clone()
    idx2[0, 20] = (idx2[0, 20] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits1, _ = model(idx)
        logits2, _ = model(idx2)
    assert torch.equal(logits1[:, :20], logits2[:, :20]), "future token leaked into past"
    assert not torch.equal(logits1[:, 20:], logits2[:, 20:])


def test_weight_tying():
    model = Ember(TINY)
    assert model.lm_head.weight is model.tok_emb.weight


def test_generate_shapes_and_determinism():
    torch.manual_seed(0)
    model = Ember(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (2, 5))
    out = model.generate(prompt, max_new_tokens=7, temperature=0.0)
    assert out.shape == (2, 12)
    out2 = model.generate(prompt, max_new_tokens=7, temperature=0.0)
    assert torch.equal(out, out2)  # greedy is deterministic
    out3 = model.generate(prompt, max_new_tokens=7, temperature=1.0, top_k=10, top_p=0.9)
    assert out3.shape == (2, 12)
