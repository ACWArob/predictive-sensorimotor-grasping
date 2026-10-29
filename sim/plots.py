"""Result figures. Baselines in neutral grey, our modes in the reference categorical order."""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator   # noqa: E402


def _logx(ax, ticks):
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"deligrasp": "#9a9893", "expforce": "#52514e",
          "predictor": "#2a78d6", "predictor+update": "#eb6834",
          "full": "#1baf7a", "full-nogate": "#eda100"}
STYLE = {"deligrasp": "--", "expforce": "--"}
LABEL = {"deligrasp": "DeliGrasp-like (semantic prior)", "expforce": "Exp-Force-like (retrieval)",
         "predictor": "Predictor only", "predictor+update": "Predictor + one-shot update",
         "full": "Full (+ value head, gated)", "full-nogate": "Full, no gate"}
METHODS = list(COLORS)


def _ax(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fig_deceptive(rows, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    rounds = np.arange(5)
    for m in METHODS:
        rs = _sel(rows, method=m, exp="A_deceptive")
        clean = [np.mean([r["clean"] for r in rs if r["trial"] == t]) for t in rounds]
        ratio = [np.median([r["force_ratio"] for r in rs if r["trial"] == t]) for t in rounds]
        kw = dict(color=COLORS[m], ls=STYLE.get(m, "-"), lw=2, marker="o", ms=5, label=LABEL[m])
        a1.plot(rounds + 1, clean, **kw)
        a2.plot(rounds + 1, ratio, **kw)
    _ax(a1, "Deceptive objects: clean success by grasp number", "share lifted & held, no slip")
    a1.set_ylim(-0.03, 1.03); a1.set_xticks(rounds + 1); a1.set_xlabel("grasp of the same object", color=INK2, fontsize=9)
    _ax(a2, "Initial grip vs what the object needed", "commanded / needed (median, log)")
    a2.set_yscale("log"); _logx(a2, [0.5, 1, 2, 4, 8]); a2.axhspan(1.0, 2.5, color="#1baf7a", alpha=0.08, lw=0)
    a2.axhline(1.0, color=INK2, lw=0.8)
    a2.text(5.05, 1.55, "appropriate\n(1x to 2.5x)", fontsize=8, color=INK2, va="center")
    a2.set_xticks(rounds + 1); a2.set_xlabel("grasp of the same object", color=INK2, fontsize=9)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(out, "fig_deceptive.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_noleak(rows, out):
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=SURF)
    trials = np.arange(9)
    for m in ("deligrasp", "expforce", "predictor", "predictor+update"):
        rs = _sel(rows, method=m, exp="B_noleak")
        y = [np.median([r["force_ratio"] for r in rs if r["trial"] == t]) for t in trials]
        ax.plot(trials + 1, y, color=COLORS[m], ls=STYLE.get(m, "-"), lw=2, marker="o", ms=5, label=LABEL[m])
    _ax(ax, "Identical-looking bottles: does a correction leak?", "initial grip / needed (median, log)")
    ax.set_yscale("log"); _logx(ax, [0.5, 1, 2, 4, 8, 16]); ax.set_ylim(0.35, 30)
    ax.axvspan(3.5, 6.5, color=GRID, alpha=0.6, lw=0)
    for x, t in ((2, "bottle empty"), (5, "bottle full"), (8, "empty again")):
        ax.text(x, 24, t, ha="center", fontsize=9, color=INK2)
    ax.axhspan(1.0, 2.5, color="#1baf7a", alpha=0.08, lw=0); ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xticks(trials + 1); ax.set_xlabel("grasp in sequence", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_noleak.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_ablation(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURF, sharey=True)
    metrics = (("success", "Lifted & held"), ("clean", "...without a slip"), ("appropriate", "Initial grip appropriate"))
    test = [r for r in rows if r["exp"] in ("A_deceptive", "C_heldout")]
    y = np.arange(len(METHODS))[::-1]
    for ax, (k, t) in zip(axes, metrics):
        v = [np.mean([r[k] for r in test if r["method"] == m]) for m in METHODS]
        ax.barh(y, v, color=[COLORS[m] for m in METHODS], height=0.6)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.02, yi, f"{vi:.0%}", va="center", fontsize=8.5, color=INK)
        _ax(ax, t, "")
        ax.set_xlim(0, 1.15); ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[m] for m in METHODS], color=INK, fontsize=8.5)
    fig.suptitle("All test grasps (30 unseen + 30 deceptive per seed)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ablation.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


def make_plots(rows, out):
    fig_deceptive(rows, out)
    fig_noleak(rows, out)
    fig_ablation(rows, out)


if __name__ == "__main__":
    import csv
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
    rows = list(csv.DictReader(open(os.path.join(out, "grasps.csv"))))
    for r in rows:
        for k in ("trial", "success", "clean", "appropriate", "crushed"):
            r[k] = int(r[k])
        r["force_ratio"] = float(r["force_ratio"])
    make_plots(rows, out)
