"""Draw docs/assets/fig_versions.png: v1 vs v2 vs v3 grasp pipelines + the paired test harness."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets", "fig_versions.png")
GREY, BLUE, ORANGE, GREEN, INK, MUTED, BG = "#9a978f", "#2a7ae2", "#ee6a35", "#1aa774", "#111111", "#555555", "#fcfbf9"


def box(ax, x, y, w, h, title, lines, col, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", lw=2.2, ec=col, fc=fill))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12.5, weight="bold", color=col if col != GREY else INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - 0.34 * i, ln, ha="center", va="center", fontsize=9.6, color=MUTED)


def arrow(ax, a, b, col="#555555", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=1.8, color=col, linestyle=ls))


def row(ax, y, label, sub, stages, col):
    ax.text(0.2, y + 0.95, label, fontsize=15, weight="bold", color=col, va="center")
    ax.text(0.2, y + 0.45, sub, fontsize=10, color=MUTED, va="center")
    x, w, h, gap = 2.75, 2.95, 1.55, 0.4
    for i, (t, ls, c) in enumerate(stages):
        box(ax, x, y, w, h, t, ls, c, "#f3f2ef" if c == GREY else {BLUE: "#eaf2fd", ORANGE: "#fdeee7", GREEN: "#e6f6ef"}.get(c, "#fff"))
        if i:
            arrow(ax, (x - gap + 0.02, y + h / 2), (x - 0.02, y + h / 2))
        x += w + gap


fig, ax = plt.subplots(figsize=(19.5, 12.2), dpi=110)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 19.5); ax.set_ylim(0, 12.2); ax.axis("off")
ax.text(0.2, 11.75, "AutoGrasp v1 → v2 → v3: what changes at each step of one grasp", fontsize=19, weight="bold", color=INK)
ax.text(0.2, 11.3, "Grey = same as current AutoGrasp.  Blue = learned offline.  Orange = new online loop.  Green = new in v3.",
        fontsize=11.5, color=MUTED)

S = [("Before grasp", None), ("Close", None), ("Lift-off", None), ("Carry / hold", None), ("After grasp", None)]
for i, (t, _) in enumerate(S):
    ax.text(2.75 + 3.35 * i + 1.475, 10.75, t, ha="center", fontsize=12, weight="bold", color=INK)

row(ax, 8.5, "v1", "current AutoGrasp\n(DeliGrasp-style)", [
    ("Semantic prior", ["LLM guesses mass, μ", "from what it looks like"], GREY),
    ("Fixed grip", ["grip = m·g / 2μ × 1.5", "set once at contact"], GREY),
    ("Nothing", ["no check against", "what the hand feels"], GREY),
    ("Slip reflex", ["slip → grip × 1.4"], GREY),
    ("Forget", ["next grasp of the same", "object starts from scratch"], GREY)], MUTED)

row(ax, 5.9, "v2", "PSG: predict,\ncompare, correct", [
    ("Forward model", ["predicts lift-off load,", "slip ratio, + uncertainty σ"], BLUE),
    ("Grip from prediction", ["grip = load × ratio", "× margin(σ)"], BLUE),
    ("Compare", ["felt load vs predicted:", "> 2σ off → raise grip"], ORANGE),
    ("Slip reflex", ["same × 1.4 reflex"], GREY),
    ("Correct once", ["per-object Kalman offset;", "crush → force ceiling"], ORANGE)], BLUE)

row(ax, 3.3, "v3", "PSG + grip-load\ncoupling", [
    ("Forward model", ["same as v2"], BLUE),
    ("Close gently", ["start at 0.35× planned", "grip (floor 0.3 N)"], GREEN),
    ("Couple grip to load", ["grip tracks felt load", "× ratio × 1.4 (Johansson)"], GREEN),
    ("Slip reflex", ["same × 1.4 reflex"], GREY),
    ("Correct once", ["same as v2"], ORANGE)], GREEN)

# harness strip
ax.add_patch(FancyBboxPatch((0.2, 0.25), 19.0, 2.55, boxstyle="round,pad=0,rounding_size=0.12", lw=1.5, ec="#dddddd", fc="#f6f5f2"))
ax.text(0.55, 2.45, "Test harness (harness/compare_autograsp.py): every version gets the same objects, order, seeds, percept noise and physics",
        fontsize=12.5, weight="bold", color=INK)
H = [("Phase-0 log", "48 honest objects, ~240 grasps\n→ trains the forward model"),
     ("5 suites", "honest · deceptive · look-alike\nfragile · repeat"),
     ("MuJoCo", "box jaw (sim/env.py) or the\nMAGPIE 4-bar model (env_magpie.py)"),
     ("Paired metrics", "success, crushes, drops,\nexcess force = grip ÷ needed"),
     ("Pre-registered checks", "5 pass/fail rules, bootstrap CIs\n→ report.md, trials.csv, .xlsx")]
for i, (t, d) in enumerate(H):
    x = 0.55 + 3.75 * i
    ax.text(x, 1.85, t, fontsize=11.5, weight="bold", color=INK)
    ax.text(x, 1.2, d, fontsize=10, color=MUTED, va="center", linespacing=1.5)
    if i:
        arrow(ax, (x - 0.5, 1.85), (x - 0.15, 1.85), "#888888")
fig.savefig(OUT, facecolor=BG, bbox_inches="tight")
print(OUT)
