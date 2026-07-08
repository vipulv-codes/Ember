"""Generate text from a trained checkpoint.

Usage:
    python src/sample.py --ckpt checkpoints/ember_30m_hero/best.pt \
        --prompt "Once upon a time" --max-new 200 --temperature 0.8 --top-p 0.95
"""

import argparse
import json
from pathlib import Path

import torch

from model import Ember, preset_config
from tokenizer import EOT, CharTokenizer, load_bpe

ROOT = Path(__file__).resolve().parents[1]


def load_model(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    variant_keys = ["pos_encoding", "norm_type", "norm_placement", "ffn_type"]
    model_cfg = preset_config(
        cfg["preset"],
        vocab_size=cfg["vocab_size"],
        ctx_len=cfg["ctx_len"],
        **{k: cfg[k] for k in variant_keys if k in cfg},
        # a run may override the preset's FFN width (e.g. param-matched GELU)
        **({"ffn_hidden": cfg["ffn_hidden"]} if cfg.get("ffn_hidden") else {}),
    )
    model = Ember(model_cfg).to(device).eval()
    model.load_state_dict(state["model"])
    return model, cfg, state["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--char-tokenizer", default=None, help="path to char json (ladder-3 runs)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, cfg, step = load_model(args.ckpt, device)
    print(f"loaded {cfg['run_name']} @ step {step} on {device}\n")

    if args.char_tokenizer:
        tok = CharTokenizer.load(args.char_tokenizer)
        encode = lambda s: tok.encode(s)
        decode = lambda ids: tok.decode(ids)
        eos_id = None
    else:
        tok = load_bpe(ROOT / "data" / "tokenizer.json")
        encode = lambda s: tok.encode(s).ids
        decode = lambda ids: tok.decode(ids)
        eos_id = tok.token_to_id(EOT)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=eos_id,
        )
        print(f"--- sample {i + 1} ---")
        print(decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
