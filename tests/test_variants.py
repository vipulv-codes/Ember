"""Ablation-variant sanity: each axis forward-passes, respects causality,
and matches its closed-form parameter count."""

import pytest
import torch

from model import Ember, EmberConfig, preset_config

BASE = dict(vocab_size=101, ctx_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)

VARIANTS = {
    "learned_pos": dict(pos_encoding="learned"),
    "layernorm": dict(norm_type="layernorm"),
    "postnorm": dict(norm_placement="post"),
    "gelu": dict(ffn_type="gelu", ffn_hidden=264),  # 1.5x to match swiglu params
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_forward_and_causality(name):
    torch.manual_seed(0)
    cfg = EmberConfig(**{**BASE, **VARIANTS[name]})
    model = Ember(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 32, cfg.vocab_size) and torch.isfinite(loss)
    # causality still holds under every variant
    idx2 = idx.clone()
    idx2[0, 20] = (idx2[0, 20] + 1) % cfg.vocab_size
    with torch.no_grad():
        l1, _ = model(idx)
        l2, _ = model(idx2)
    assert torch.equal(l1[:, :20], l2[:, :20])


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_param_count(name):
    cfg = EmberConfig(**{**BASE, **VARIANTS[name]})
    model = Ember(cfg)
    unique = sum(t.numel() for t in {p.data_ptr(): p for p in model.parameters()}.values())
    assert unique == cfg.n_params()


def test_gelu_matches_swiglu_params():
    swiglu = EmberConfig(**BASE)
    gelu = EmberConfig(**{**BASE, "ffn_type": "gelu", "ffn_hidden": 264})
    assert gelu.n_params() == swiglu.n_params()


def test_deep_preset_matches_wide_params():
    wide = preset_config("ember-12m")
    deep = preset_config("ember-12m-deep")
    assert abs(deep.n_params() - wide.n_params()) / wide.n_params() < 0.005


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_cache_equivalence(name):
    """KV cache must stay exact under every architecture variant."""
    torch.manual_seed(0)
    cfg = EmberConfig(**{**BASE, **VARIANTS[name]})
    model = Ember(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 10))
    a = model.generate(prompt, max_new_tokens=20, temperature=0.0, use_cache=True)
    b = model.generate(prompt, max_new_tokens=20, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_generate_restores_training_mode():
    torch.manual_seed(0)
    model = Ember(EmberConfig(**BASE)).train()
    model.generate(torch.randint(0, 101, (1, 4)), max_new_tokens=3, temperature=0.0)
    assert model.training  # eval() during generation must not leak out
