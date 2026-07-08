"""The flagship plot (plan section 3A): val loss vs params and vs training
FLOPs for the four sizes, each trained to the same fixed token budget with a
fully annealed cosine. Log-log, with a fitted power law L = A * N^b.

Usage: python scripts/plot_scaling.py   (after the scale_* runs finish)
"""

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import preset_config

ROOT = Path(__file__).resolve().parents[1]
# LR-consistent set: all four at peak LR 6e-4 (the LR sweep, experiments
# #13-14, showed the original 3e-4-for-big-models policy costs ~0.2 nats
# at this 953-step regime and bent the curve).
RUNS = ["scale_5m", "scale_12m", "ablate_30m_lr6e4", "ablate_60m_lr6e4"]
# Seed sigma measured over 3 seeds (1337/1338/1339) at the two small sizes
# (log #21-22); larger sizes are expected to sit at or below this noise level.
SEED_SIGMA = {"scale_5m": 0.0050, "scale_12m": 0.0051}
PRESETS_FOR = {"scale_5m": "ember-5m", "scale_12m": "ember-12m",
               "ablate_30m_lr6e4": "ember-30m", "ablate_60m_lr6e4": "ember-60m"}
LABELS = ["ember-5M", "ember-12M", "ember-30M", "ember-60M"]

SERIES = "#2a78d6"  # categorical slot 1
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def final_val(run: str) -> tuple[float, int]:
    path = ROOT / "experiments" / "runs" / f"{run}.csv"
    if not path.exists():
        sys.exit(f"missing {path} — has the {run} run finished?")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["val_loss"]]
    last = rows[-1]
    return float(last["val_loss"]), int(last["tokens"])


def main():
    n_params, losses, tokens = [], [], []
    for run in RUNS:
        n = preset_config(PRESETS_FOR[run]).n_params()
        loss, toks = final_val(run)
        n_params.append(n)
        losses.append(loss)
        tokens.append(toks)
        print(f"{run}: N={n / 1e6:.1f}M  D={toks / 1e6:.0f}M  val={loss:.4f}")

    N = np.array(n_params, dtype=float)
    L = np.array(losses)
    D = float(np.mean(tokens))
    C = 6.0 * N * D  # training FLOPs approx

    # Power-law fit in log space: log L = log A + b log N
    b, logA = np.polyfit(np.log(N), np.log(L), 1)
    fit = lambda n: np.exp(logA) * n**b

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, x, xlabel in [
        (axes[0], N, "parameters"),
        (axes[1], C, "training FLOPs (≈ 6·N·D)"),
    ]:
        ax.set_facecolor(SURFACE)
        xs = np.geomspace(x.min() * 0.8, x.max() * 1.25, 100)
        ns = np.geomspace(N.min() * 0.8, N.max() * 1.25, 100)
        ax.plot(xs, fit(ns), "--", color=INK_2, linewidth=1.5, zorder=2)
        measured = [i for i, r in enumerate(RUNS) if r in SEED_SIGMA]
        ax.errorbar(
            [x[i] for i in measured], [L[i] for i in measured],
            yerr=[SEED_SIGMA[RUNS[i]] for i in measured],
            fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=3,
        )
        ax.scatter(x, L, s=90, color=SERIES, zorder=4)
        for xi, li, lab in zip(x, L, LABELS):
            ax.annotate(
                lab, (xi, li), textcoords="offset points", xytext=(8, 7),
                fontsize=9, color=INK,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel("val loss (nats)", color=INK)
        ax.grid(alpha=0.25, which="both", linewidth=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK_2, which="both")
        # log-log y ticks: show plain numbers at the data's scale
        ax.set_yticks([1.3, 1.5, 1.7, 2.0])
        ax.set_yticklabels(["1.3", "1.5", "1.7", "2.0"])
        ax.minorticks_off()

    axes[0].annotate(
        f"L ∝ N^{b:.3f}", xy=(0.06, 0.12), xycoords="axes fraction",
        fontsize=11, color=INK_2, style="italic",
    )
    axes[0].annotate(
        "±1σ over 3 seeds at 5M/12M (σ≈0.005 — bars barely exceed the markers)",
        xy=(0.06, 0.05), xycoords="axes fraction", fontsize=8, color=INK_2,
    )
    fig.suptitle(
        f"Ember scaling study — four sizes, fixed {D / 1e6:.0f}M-token budget, peak LR 6e-4",
        color=INK, fontsize=12,
    )
    fig.tight_layout()
    out = ROOT / "report" / "scaling.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"fit: L = {np.exp(logA):.2f} * N^{b:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
