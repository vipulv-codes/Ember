"""Regenerate loss curves from run CSVs.

Usage: python scripts/plot_loss.py experiments/runs/ember_30m_hero.csv [more.csv ...]
Writes a .png next to each CSV.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    tokens = [int(r["tokens"]) / 1e6 for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train, label="train loss", alpha=0.8)
    if val:
        ax.plot(*zip(*val), label="val loss", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(f"{csv_path.stem} — {tokens[-1]:.0f}M tokens")
    ax.legend()
    ax.grid(alpha=0.3)
    out = csv_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot(Path(p))
