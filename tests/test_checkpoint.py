import torch

from model import Ember, EmberConfig, preset_config
from sample import load_model

TINY = EmberConfig(vocab_size=101, ctx_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)


def test_state_dict_round_trip(tmp_path):
    torch.manual_seed(0)
    m1 = Ember(TINY).eval()
    path = tmp_path / "ckpt.pt"
    torch.save(m1.state_dict(), path)

    torch.manual_seed(999)  # different init, must be fully overwritten by load
    m2 = Ember(TINY).eval()
    m2.load_state_dict(torch.load(path, weights_only=True))

    idx = torch.randint(0, TINY.vocab_size, (2, 32))
    with torch.no_grad():
        logits1, _ = m1(idx)
        logits2, _ = m2(idx)
    assert torch.equal(logits1, logits2)
    assert m2.lm_head.weight is m2.tok_emb.weight  # tying survives load


def test_load_model_respects_ffn_hidden_override(tmp_path):
    """Regression: a run that overrides the preset's FFN width (param-matched
    GELU ablation) must reconstruct with that width, not the preset default."""
    torch.manual_seed(0)
    cfg = preset_config("ember-5m", vocab_size=101, ctx_len=64,
                        ffn_type="gelu", ffn_hidden=1056)
    m1 = Ember(cfg).eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({
        "model": m1.state_dict(),
        "step": 7,
        "config": {"run_name": "t", "preset": "ember-5m", "vocab_size": 101,
                   "ctx_len": 64, "ffn_type": "gelu", "ffn_hidden": 1056},
    }, ckpt)
    m2, _, step = load_model(str(ckpt), "cpu")
    assert step == 7
    idx = torch.randint(0, 101, (1, 16))
    with torch.no_grad():
        l1, _ = m1(idx)
        l2, _ = m2(idx)
    assert torch.equal(l1, l2)
