import pytest
import torch

from model import Ember, EmberConfig, KVCache

TINY = EmberConfig(vocab_size=101, ctx_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)


def test_greedy_cache_equals_no_cache():
    """The plan's non-negotiable: cached greedy decoding must match the
    recompute-everything path token-for-token."""
    torch.manual_seed(0)
    model = Ember(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (2, 10))
    out_cache = model.generate(prompt, max_new_tokens=40, temperature=0.0, use_cache=True)
    out_plain = model.generate(prompt, max_new_tokens=40, temperature=0.0, use_cache=False)
    assert torch.equal(out_cache, out_plain)


@pytest.mark.parametrize("impl", ["sdpa", "naive"])
def test_stepwise_logits_match_full_forward(impl):
    """Prefill + N single-token decodes must reproduce the full-sequence logits."""
    torch.manual_seed(0)
    cfg = EmberConfig(**{**TINY.__dict__, "attn_impl": impl})
    model = Ember(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(cfg, 2)
        pre, _ = model(idx[:, :8], cache=cache)
        steps = [pre[:, -1]]
        for t in range(8, 24):
            step, _ = model(idx[:, t : t + 1], cache=cache)
            steps.append(step[:, -1])
    stepwise = torch.stack(steps, dim=1)  # logits at positions 7..23
    torch.testing.assert_close(stepwise, full[:, 7:], atol=1e-4, rtol=1e-4)


def test_chunked_append_matches_full_forward():
    """The offset-mask path: appending a multi-token chunk onto a warm cache."""
    torch.manual_seed(0)
    model = Ember(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 20))
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(TINY, 1)
        model(idx[:, :12], cache=cache)
        chunk, _ = model(idx[:, 12:], cache=cache)  # 8 tokens onto pos=12
    torch.testing.assert_close(chunk, full[:, 12:], atol=1e-4, rtol=1e-4)


def test_cache_memory_formula():
    """bytes == 2(K,V) * layers * heads * head_dim * ctx_len * batch * elem_size"""
    model = Ember(TINY).eval()
    cache = KVCache(TINY, batch_size=3)
    with torch.no_grad():
        model(torch.randint(0, TINY.vocab_size, (3, 5)), cache=cache)
    expected = 2 * TINY.n_layers * TINY.n_heads * TINY.head_dim * TINY.ctx_len * 3 * 4
    assert cache.nbytes == expected
    assert cache.pos == 5


def test_cache_overflow_asserts():
    model = Ember(TINY).eval()
    cache = KVCache(TINY, 1)
    with torch.no_grad(), pytest.raises(AssertionError, match="overflow"):
        model(torch.randint(0, TINY.vocab_size, (1, 60)), cache=cache)
        model(torch.randint(0, TINY.vocab_size, (1, 60)), cache=cache)
