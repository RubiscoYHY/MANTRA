"""
experiments/chi_square_combined.py

Combine results from ALL available date rounds and perform a joint
chi-square / Fisher's exact test with higher statistical power.

Usage:
    python experiments/chi_square_combined.py
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats

FIGURES_DIR = Path("experiments/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "BUY":  "#2ecc71",
    "SELL": "#e74c3c",
    "HOLD": "#95a5a6",
    "OLD":  "#e67e22",
    "NEW":  "#3498db",
}
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

_NORM = {
    "OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL", "STRONG BUY": "BUY",
    "STRONG SELL": "SELL", "NEUTRAL": "HOLD",
}
def _normalise(sig: str) -> str:
    return _NORM.get(sig.upper().strip(), sig.upper().strip())

# ── Load all rounds ────────────────────────────────────────────────────────────
old_files = sorted(glob.glob("experiments/results_old_arch_*.json"))
new_files = sorted(glob.glob("experiments/results_new_arch_*.json"))

print(f"Found {len(old_files)} OLD round(s): {[Path(f).name for f in old_files]}")
print(f"Found {len(new_files)} NEW round(s): {[Path(f).name for f in new_files]}")

def load_all(files, arch_label):
    rows, signals = [], []
    for path in files:
        date_tag = Path(path).stem.split("_")[-1]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            if "error" not in row:
                sig = _normalise(row.get("signal", "UNKNOWN"))
                signals.append(sig)
                rows.append({**row, "signal": sig, "date": date_tag, "arch": arch_label})
            else:
                print(f"  [{arch_label}] Skip {row.get('ticker','?')} ({date_tag}): {row['error'][:60]}")
    return rows, signals

old_rows, old_signals = load_all(old_files, "OLD")
new_rows, new_signals = load_all(new_files, "NEW")

CATS = ["BUY", "SELL", "HOLD"]
old_counts = {c: old_signals.count(c) for c in CATS}
new_counts = {c: new_signals.count(c) for c in CATS}
n_old, n_new = len(old_signals), len(new_signals)

# ── Contingency table ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"COMBINED CONTINGENCY TABLE  (n_old={n_old}, n_new={n_new})")
print(f"{'='*65}")
old_row = [old_counts[c] for c in CATS]
new_row = [new_counts[c] for c in CATS]
col_totals = [old_counts[c]+new_counts[c] for c in CATS]
print(f"{'':14s}" + "".join(f"{c:>8s}" for c in CATS) + f"{'TOTAL':>8s}")
print("-"*65)
print(f"{'OLD (no judge)':14s}" + "".join(f"{v:>8d}" for v in old_row) + f"{n_old:>8d}")
print(f"{'NEW (judge)':14s}"    + "".join(f"{v:>8d}" for v in new_row) + f"{n_new:>8d}")
print("-"*65)
print(f"{'TOTAL':14s}" + "".join(f"{v:>8d}" for v in col_totals) + f"{n_old+n_new:>8d}")

# ── Chi-square (2×3) ───────────────────────────────────────────────────────────
observed = np.array([old_row, new_row])
valid_cols = [i for i, t in enumerate(col_totals) if t > 0]
obs_f = observed[:, valid_cols]
labels_f = [CATS[i] for i in valid_cols]
chi2, p_chi2, dof, expected = stats.chi2_contingency(obs_f, correction=False)

print(f"\n{'='*65}")
print("CHI-SQUARE TEST  (combined rounds)")
print(f"{'='*65}")
print(f"  Categories : {labels_f}")
print(f"  Chi²       : {chi2:.4f}")
print(f"  df         : {dof}")
print(f"  p-value    : {p_chi2:.4f}  {'*** SIGNIFICANT' if p_chi2 < 0.05 else '(not significant)'}")

# ── Fisher's exact: SELL vs non-SELL ──────────────────────────────────────────
old_sell, new_sell = old_counts["SELL"], new_counts["SELL"]
old_ns,   new_ns   = n_old - old_sell,   n_new - new_sell
odds_ratio, p_fisher = stats.fisher_exact(
    [[old_sell, old_ns], [new_sell, new_ns]], alternative="two-sided"
)

p_old_sell = old_sell / n_old if n_old else 0
p_new_sell = new_sell / n_new if n_new else 0

print(f"\n{'='*65}")
print("FISHER'S EXACT TEST  (SELL vs non-SELL, combined)")
print(f"{'='*65}")
print(f"  OLD  SELL={old_sell}/{n_old} ({p_old_sell:.0%})   non-SELL={old_ns}")
print(f"  NEW  SELL={new_sell}/{n_new} ({p_new_sell:.0%})   non-SELL={new_ns}")
print(f"  Odds ratio : {odds_ratio:.4f}")
print(f"  p-value    : {p_fisher:.4f}  {'*** SIGNIFICANT' if p_fisher < 0.05 else '(not significant)'}")
print(f"  SELL diff  : {p_old_sell - p_new_sell:+.1%}  (OLD − NEW)")

# ── Per-date breakdown ─────────────────────────────────────────────────────────
dates = sorted({r["date"] for r in old_rows + new_rows})
print(f"\n{'='*65}")
print("PER-DATE BREAKDOWN")
print(f"{'='*65}")
for d in dates:
    o_sigs = [r["signal"] for r in old_rows if r["date"] == d]
    n_sigs = [r["signal"] for r in new_rows if r["date"] == d]
    print(f"\n  {d}:")
    print(f"    OLD  BUY={o_sigs.count('BUY')} SELL={o_sigs.count('SELL')} HOLD={o_sigs.count('HOLD')}")
    print(f"    NEW  BUY={n_sigs.count('BUY')} SELL={n_sigs.count('SELL')} HOLD={n_sigs.count('HOLD')}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure A: Grouped bar with per-date stacking ───────────────────────────────
fig, axes = plt.subplots(1, len(dates)+1, figsize=(4*(len(dates)+1), 4.8),
                         gridspec_kw={"width_ratios": [1]*len(dates) + [1.3]})

sig_label_text = (f"Combined: χ²={chi2:.2f}, p={p_chi2:.3f}"
                  + (" *" if p_chi2 < 0.05 else " n.s."))

for ax_i, d in enumerate(dates):
    ax = axes[ax_i]
    o_sigs = [r["signal"] for r in old_rows if r["date"] == d]
    n_sigs = [r["signal"] for r in new_rows if r["date"] == d]
    o_cnt  = [o_sigs.count(c) for c in CATS]
    n_cnt  = [n_sigs.count(c) for c in CATS]
    x = np.arange(len(CATS))
    w = 0.35
    bars_o = ax.bar(x - w/2, o_cnt, w, color=[PALETTE[c] for c in CATS],
                    alpha=0.7, edgecolor="white", hatch="///")
    bars_n = ax.bar(x + w/2, n_cnt, w, color=[PALETTE[c] for c in CATS],
                    alpha=1.0, edgecolor="white")
    for bar in list(bars_o) + list(bars_n):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.05, str(int(h)),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=10)
    ax.set_title(f"Date: {d}", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(max(o_cnt), max(n_cnt)) + 2.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel("Count" if ax_i == 0 else "")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Combined panel
ax_c = axes[-1]
x = np.arange(len(CATS)); w = 0.35
bars_o = ax_c.bar(x - w/2, old_row, w, color=[PALETTE[c] for c in CATS],
                  alpha=0.7, edgecolor="white", hatch="///")
bars_n = ax_c.bar(x + w/2, new_row, w, color=[PALETTE[c] for c in CATS],
                  alpha=1.0, edgecolor="white")
for bar in list(bars_o) + list(bars_n):
    h = bar.get_height()
    if h > 0:
        ax_c.text(bar.get_x()+bar.get_width()/2, h+0.05, str(int(h)),
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_c.set_xticks(x); ax_c.set_xticklabels(CATS, fontsize=10)
ax_c.set_title("Combined (all dates)", fontsize=10, fontweight="bold")
ax_c.set_ylim(0, max(max(old_row), max(new_row)) + 2.2)
ax_c.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
ax_c.text(0.03, 0.97, sig_label_text, transform=ax_c.transAxes,
          fontsize=8, va="top", color="#555",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ccc"))

legend_handles = [
    mpatches.Patch(facecolor="#aaa", hatch="///", edgecolor="white", label="Old arch"),
    mpatches.Patch(facecolor="#aaa", edgecolor="white",              label="New arch"),
    mpatches.Patch(facecolor=PALETTE["BUY"],  label="BUY"),
    mpatches.Patch(facecolor=PALETTE["SELL"], label="SELL"),
    mpatches.Patch(facecolor=PALETTE["HOLD"], label="HOLD"),
]
fig.legend(handles=legend_handles, fontsize=9, loc="lower center",
           ncol=5, framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Signal Distribution by Architecture — All Rounds",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
path_a = FIGURES_DIR / "figA_combined_distribution.png"
fig.savefig(path_a, dpi=180, bbox_inches="tight")
print(f"\n[Figure A saved] {path_a}")

# ── Figure B: SELL rate across dates + combined ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5 + len(dates)*0.8, 4.2))

x_labels = dates + ["Combined"]
sell_old_by = []
sell_new_by = []
for d in dates:
    o_s = [r["signal"] for r in old_rows if r["date"] == d]
    n_s = [r["signal"] for r in new_rows if r["date"] == d]
    sell_old_by.append(o_s.count("SELL") / len(o_s) if o_s else 0)
    sell_new_by.append(n_s.count("SELL") / len(n_s) if n_s else 0)
sell_old_by.append(p_old_sell)
sell_new_by.append(p_new_sell)

x = np.arange(len(x_labels)); w = 0.35
b1 = ax2.bar(x - w/2, sell_old_by, w, color=PALETTE["OLD"], alpha=0.85,
             label="Old arch (Bear last)", edgecolor="white")
b2 = ax2.bar(x + w/2, sell_new_by, w, color=PALETTE["NEW"], alpha=0.85,
             label="New arch (Judge)", edgecolor="white")
for bar, rate in zip(list(b1)+list(b2), sell_old_by+sell_new_by):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{rate:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(x_labels, fontsize=11)
ax2.set_ylabel("SELL rate", fontsize=11)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.set_ylim(0, max(sell_old_by + sell_new_by) + 0.22)
ax2.set_title(
    f"SELL Rate: Old vs New Architecture\n"
    f"Fisher (combined): OR={odds_ratio:.2f}, p={p_fisher:.3f}"
    + (" *" if p_fisher < 0.05 else " (n.s.)"),
    fontsize=11, fontweight="bold", pad=10
)
ax2.legend(fontsize=9)
# Highlight combined bar
ax2.axvline(len(dates) - 0.5, color="#bbb", linestyle="--", linewidth=1)
ax2.text(len(dates) + 0.1, ax2.get_ylim()[1]*0.97, "combined",
         fontsize=8, color="#888", va="top")

fig2.tight_layout()
path_b = FIGURES_DIR / "figB_sell_rate_by_date.png"
fig2.savefig(path_b, dpi=180, bbox_inches="tight")
print(f"[Figure B saved] {path_b}")

plt.close("all")
print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")
