"""Render the M3 benchmark chart from experiments/benchmarks.csv.

Grouped horizontal bars: generation tokens/sec for each precision/device,
without (gray baseline) and with (blue) the KV cache. Direct value labels.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SERIES = "#2a78d6"  # KV cache
BASELINE = "#9b9a96"  # no cache (muted)
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def main():
    with open(ROOT / "experiments" / "benchmarks.csv") as f:
        rows = list(csv.DictReader(f))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    val = {(r["variant"], r["kv_cache"] == "True"): float(r["tok_per_sec"]) for r in rows}
    quality = json.loads((ROOT / "experiments" / "benchmark_quality.json").read_text())

    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(variants))
    h = 0.34
    no_cache = [val[(v, False)] for v in variants]
    cache = [val[(v, True)] for v in variants]
    b1 = ax.barh(y + h / 2 + 0.01, no_cache, height=h, color=BASELINE, label="no cache")
    b2 = ax.barh(y - h / 2 - 0.01, cache, height=h, color=SERIES, label="KV cache")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_width():.0f}",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK,
            )
    ax.set_yticks(y, variants)
    ax.invert_yaxis()
    ax.set_xlabel("generation speed (tokens/sec, greedy, 128 new tokens)", color=INK)
    ax.set_xlim(0, max(cache + no_cache) * 1.12)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_2)
    ax.set_title(
        "ember-30M single-stream generation — KV cache & quantization\n"
        f"int8 quality cost: ppl {quality['ppl_fp32']} → {quality['ppl_int8']} "
        f"(val loss +{quality['val_loss_int8'] - quality['val_loss_fp32']:.3f} nats)",
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    out = ROOT / "report" / "benchmarks.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
