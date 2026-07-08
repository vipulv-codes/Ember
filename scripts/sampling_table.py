"""Qualitative sampling table (plan section 4): the same prompt across
temperature / top-k / top-p settings, one gallery per checkpoint.

Usage: python scripts/sampling_table.py --ckpt checkpoints/ember_30m_hero/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sample import load_model
from tokenizer import EOT, load_bpe

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "One day, a little robot found a red button."

SETTINGS = [
    ("greedy (T=0)", dict(temperature=0.0)),
    ("T=0.5", dict(temperature=0.5)),
    ("T=0.8", dict(temperature=0.8)),
    ("T=0.8, top-k 40", dict(temperature=0.8, top_k=40)),
    ("T=0.8, top-p 0.95", dict(temperature=0.8, top_p=0.95)),
    ("T=1.0", dict(temperature=1.0)),
    ("T=1.3 (too hot)", dict(temperature=1.3)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=140)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, step = load_model(args.ckpt, device)
    tok = load_bpe(ROOT / "data" / "tokenizer.json")
    idx = torch.tensor([tok.encode(PROMPT).ids], dtype=torch.long, device=device)

    lines = [
        f"# Sampling settings table — {cfg['run_name']} @ step {step}",
        "",
        f"Prompt: **{PROMPT}** (seed 1337 per setting, {args.max_new} max new tokens)",
        "",
    ]
    for name, kw in SETTINGS:
        torch.manual_seed(1337)
        out = model.generate(
            idx, max_new_tokens=args.max_new, eos_id=tok.token_to_id(EOT), **kw
        )
        text = tok.decode(out[0].tolist()).replace(EOT, "").strip()
        lines += [f"## {name}", "", text, ""]
        print(f"done: {name}", flush=True)

    out_path = ROOT / "experiments" / "samples" / f"{cfg['run_name']}_sampling_table.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
