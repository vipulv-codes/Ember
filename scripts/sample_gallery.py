"""Fixed 30-prompt sample gallery for a checkpoint (plan section 3C).

Same prompts + seed every time, so galleries are comparable across
checkpoints and runs. Writes markdown to experiments/samples/.

Usage: python scripts/sample_gallery.py --ckpt checkpoints/ember_30m_hero/best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sample import load_model
from tokenizer import EOT, load_bpe

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom and his dog Max went to the park.",
    "One day, a small bird fell out of its nest.",
    "Sara found a shiny red box in the garden.",
    "The little train was very tired.",
    "Ben wanted to bake a cake for his mom.",
    "A tiny ant carried a big leaf.",
    "Mia saw a rainbow after the rain.",
    "The old cat slept on the warm windowsill.",
    "Jack lost his favorite blue ball.",
    "Once there was a fish who wanted to fly.",
    "Emma and her brother built a sandcastle.",
    "The moon looked down at the quiet town.",
    "A little fox was lost in the forest.",
    "Tim found a key under his bed.",
    "The snowman smiled at the children.",
    "Lucy planted a seed in a small pot.",
    "The brave mouse walked into the dark cave.",
    "Sam heard a strange noise in the kitchen.",
    "A butterfly landed on Anna's nose.",
    "The big red tractor would not start.",
    "Two frogs sat on a log by the pond.",
    "Grandma told a story by the fire.",
    "The little star wanted a friend.",
    "Leo drew a picture of a dragon.",
    "It was raining, so Maya stayed inside.",
    "The puppy chased its tail all morning.",
    "Nina wore her new boots in the mud.",
    "A turtle and a rabbit met on the road.",
    "The wind blew Peter's hat far away.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, step = load_model(args.ckpt, device)
    tok = load_bpe(ROOT / "data" / "tokenizer.json")
    eos_id = tok.token_to_id(EOT)

    out_path = ROOT / "experiments" / "samples" / f"{cfg['run_name']}_gallery.md"
    t0 = time.time()
    lines = [
        f"# Sample gallery — {cfg['run_name']} @ step {step}",
        "",
        f"temperature {args.temperature}, top-p {args.top_p}, seed {args.seed}, "
        f"max {args.max_new} new tokens. Prompts are fixed (scripts/sample_gallery.py).",
        "",
    ]
    for i, prompt in enumerate(PROMPTS, 1):
        torch.manual_seed(args.seed + i)  # per-prompt seed: order-independent
        idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_id=eos_id,
        )
        text = tok.decode(out[0].tolist()).replace(EOT, "").strip()
        lines += [f"## {i}. {prompt}", "", text, ""]
        print(f"[{i:2d}/30] {time.time() - t0:5.0f}s", flush=True)

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
