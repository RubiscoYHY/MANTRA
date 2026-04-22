"""
experiments/factorial_analysis.py

Full 2x2x2 factorial analysis of the bias experiment.

Dimensions:
  Architecture : old (Bear-last) vs new (Judge-mediated)
  Time         : 2026-03-24 (volatile/uncertain) vs 2026-04-16 (bull market)
  Ticker group : original (AAPL, MSFT …) vs new (AVGO, TSM …)

8 cells x n=10 each = 80 observations total.
"""

import json, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

FIGURES_DIR = Path("experiments/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#95a5a6",
           "old": "#e67e22",  "new": "#3498db"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

_NORM = {"OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL", "STRONG BUY": "BUY",
         "STRONG SELL": "SELL", "NEUTRAL": "HOLD"}
def _n(s): return _NORM.get(s.upper().strip(), s.upper().strip())

# ── File map: (arch, group, date) → path ──────────────────────────────────────
FILE_MAP = {
    ("old", "original", "2026-03-24"): "experiments/results_old_arch_2026-03-24.json",
    ("old", "original", "2026-04-16"): "experiments/results_old_arch_original_2026-04-16.json",
    ("new", "original", "2026-03-24"): "experiments/results_new_arch_2026-03-24.json",
    ("new", "original", "2026-04-16"): "experiments/results_new_arch_original_2026-04-16.json",
    ("old", "new",      "2026-03-24"): "experiments/results_old_arch_new_2026-03-24.json",
    ("old", "new",      "2026-04-16"): "experiments/results_old_arch_new_2026-04-16.json",
    ("new", "new",      "2026-03-24"): "experiments/results_new_arch_new_2026-03-24.json",
    ("new", "new",      "2026-04-16"): "experiments/results_new_arch_new_2026-04-16.json",
}

ARCHS  = ["old", "new"]
GROUPS = ["original", "new"]
DATES  = ["2026-03-24", "2026-04-16"]
CATS   = ["BUY", "SELL", "HOLD"]

# Load all cells
CELLS = {}
for (arch, group, date), path in FILE_MAP.items():
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    sigs = [_n(r["signal"]) for r in rows if "error" not in r]
    CELLS[(arch, group, date)] = sigs

# ── Helper ─────────────────────────────────────────────────────────────────────
def sell_rate(sigs): return sigs.count("SELL") / len(sigs) if sigs else 0
def counts(sigs):    return [sigs.count(c) for c in CATS]

# ── Print full 2x2x2 table ─────────────────────────────────────────────────────
print("\n" + "="*72)
print("FULL 2×2×2 FACTORIAL TABLE — SELL RATE PER CELL")
print("="*72)
print(f"{'':30s} {'Old arch':>12s} {'New arch':>12s} {'Diff':>8s}")
print("-"*72)
for group in GROUPS:
    for date in DATES:
        o = CELLS[("old", group, date)]
        n = CELLS[("new", group, date)]
        po, pn = sell_rate(o), sell_rate(n)
        label = f"  {group:8s} | {date}"
        print(f"{label:30s} {po:>11.0%}  {pn:>11.0%}  {po-pn:>+7.0%}")
    print()

# ── Marginal effects ───────────────────────────────────────────────────────────
all_old = sum([CELLS[("old", g, d)] for g in GROUPS for d in DATES], [])
all_new = sum([CELLS[("new", g, d)] for g in GROUPS for d in DATES], [])
bull_old = sum([CELLS[("old", g, "2026-04-16")] for g in GROUPS], [])
bull_new = sum([CELLS[("new", g, "2026-04-16")] for g in GROUPS], [])
vol_old  = sum([CELLS[("old", g, "2026-03-24")] for g in GROUPS], [])
vol_new  = sum([CELLS[("new", g, "2026-03-24")] for g in GROUPS], [])

print("="*72)
print("MARGINAL SELL RATES")
print("="*72)
print(f"  Overall   — OLD: {sell_rate(all_old):.0%}  NEW: {sell_rate(all_new):.0%}  diff: {sell_rate(all_old)-sell_rate(all_new):+.0%}")
print(f"  Bull mkt  — OLD: {sell_rate(bull_old):.0%}  NEW: {sell_rate(bull_new):.0%}  diff: {sell_rate(bull_old)-sell_rate(bull_new):+.0%}")
print(f"  Volatile  — OLD: {sell_rate(vol_old):.0%}  NEW: {sell_rate(vol_new):.0%}  diff: {sell_rate(vol_old)-sell_rate(vol_new):+.0%}")

# ── Primary test: arch effect (40 old vs 40 new, all cells) ──────────────────
old_cnt = counts(all_old)
new_cnt = counts(all_new)
obs_arch = np.array([old_cnt, new_cnt])
vcols = [i for i,t in enumerate([o+n for o,n in zip(old_cnt,new_cnt)]) if t>0]
chi2_arch, p_arch, dof_arch, _ = stats.chi2_contingency(obs_arch[:,vcols], correction=False)
OR_arch, p_fish_arch = stats.fisher_exact(
    [[old_cnt[1], sum(old_cnt)-old_cnt[1]],
     [new_cnt[1], sum(new_cnt)-new_cnt[1]]], alternative="two-sided")

print(f"\n{'='*72}")
print("PRIMARY TEST: Architecture effect  (n=40 per arch, all cells pooled)")
print(f"{'='*72}")
print(f"  OLD  BUY={old_cnt[0]} SELL={old_cnt[1]} HOLD={old_cnt[2]}  SELL={sell_rate(all_old):.0%}")
print(f"  NEW  BUY={new_cnt[0]} SELL={new_cnt[1]} HOLD={new_cnt[2]}  SELL={sell_rate(all_new):.0%}")
print(f"  Chi²({dof_arch}) = {chi2_arch:.3f}   p = {p_arch:.4f}  {'*** SIGNIFICANT' if p_arch<0.05 else '(not significant)'}")
print(f"  Fisher SELL: OR={OR_arch:.3f}  p={p_fish_arch:.4f}  {'*** SIGNIFICANT' if p_fish_arch<0.05 else '(not significant)'}")

# ── Arch effect by date (interaction) ─────────────────────────────────────────
print(f"\n{'='*72}")
print("ARCH × TIME INTERACTION")
print(f"{'='*72}")
for date, label in [("2026-03-24","Volatile (3-24)"),("2026-04-16","Bull (4-16)")]:
    o = sum([CELLS[("old",g,date)] for g in GROUPS],[])
    n = sum([CELLS[("new",g,date)] for g in GROUPS],[])
    oc, nc = counts(o), counts(n)
    obs = np.array([oc, nc])
    vc = [i for i,t in enumerate([a+b for a,b in zip(oc,nc)]) if t>0]
    c2, pv, dof, _ = stats.chi2_contingency(obs[:,vc], correction=False)
    print(f"  {label:20s}  Chi²={c2:.3f}  p={pv:.4f}  "
          f"OLD SELL={sell_rate(o):.0%}  NEW SELL={sell_rate(n):.0%}  "
          f"{'*' if pv<0.05 else 'n.s.'}")

# ── Arch effect by group ───────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("ARCH × TICKER GROUP INTERACTION")
print(f"{'='*72}")
for group in GROUPS:
    o = sum([CELLS[("old",group,d)] for d in DATES],[])
    n = sum([CELLS[("new",group,d)] for d in DATES],[])
    oc, nc = counts(o), counts(n)
    obs = np.array([oc, nc])
    vc = [i for i,t in enumerate([a+b for a,b in zip(oc,nc)]) if t>0]
    c2, pv, dof, _ = stats.chi2_contingency(obs[:,vc], correction=False)
    print(f"  {group:8s} group  Chi²={c2:.3f}  p={pv:.4f}  "
          f"OLD SELL={sell_rate(o):.0%}  NEW SELL={sell_rate(n):.0%}  "
          f"{'*' if pv<0.05 else 'n.s.'}")

# ── HOLD comparison (structural) ──────────────────────────────────────────────
old_hold = all_old.count("HOLD")
new_hold = all_new.count("HOLD")
OR_hold, p_hold = stats.fisher_exact(
    [[old_hold, len(all_old)-old_hold],
     [new_hold, len(all_new)-new_hold]], alternative="two-sided")
print(f"\n{'='*72}")
print("HOLD SIGNAL: Structural difference")
print(f"{'='*72}")
print(f"  OLD  HOLD={old_hold}/40  ({old_hold/40:.0%})")
print(f"  NEW  HOLD={new_hold}/40  ({new_hold/40:.0%})")
print(f"  Fisher: OR={OR_hold:.3f}  p={p_hold:.4f}  {'*** SIGNIFICANT' if p_hold<0.05 else '(not significant)'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

arch_labels  = {"old": "Old arch\n(Bear last)", "new": "New arch\n(Judge)"}
group_labels = {"original": "Original\n(AAPL…)", "new": "New\n(AVGO…)"}
date_labels  = {"2026-03-24": "3-24\n(Volatile)", "2026-04-16": "4-16\n(Bull)"}

# ── Figure 1: 2×4 grid — SELL rate heatmap ────────────────────────────────────
fig1, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
fig1.suptitle("SELL Rate — Full 2×2×2 Factorial\n(Architecture × Time × Ticker Group)",
              fontsize=13, fontweight="bold", y=1.01)

col_configs = list(itertools.product(GROUPS, DATES))
for col, (group, date) in enumerate(col_configs):
    for row, arch in enumerate(ARCHS):
        ax  = axes[row, col]
        sig = CELLS[(arch, group, date)]
        bc  = [sig.count(c) for c in CATS]
        bars = ax.bar(CATS, bc, color=[PALETTE[c] for c in CATS],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, bc):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, v+0.08, str(v),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        sr = sell_rate(sig)
        ax.set_title(f"SELL={sr:.0%}", fontsize=9, color=PALETTE["SELL"],
                     fontweight="bold", pad=3)
        ax.set_ylim(0, 11)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel(arch_labels[arch], fontsize=9, labelpad=4)
        if row == 0:
            ax.set_title(f"{group_labels[group]}\n{date_labels[date]}\nSELL={sr:.0%}",
                         fontsize=8.5, pad=3, color=PALETTE["SELL"], fontweight="bold")
        else:
            ax.set_title(f"SELL={sr:.0%}", fontsize=9, color=PALETTE["SELL"],
                         fontweight="bold", pad=3)

fig1.tight_layout()
p1 = FIGURES_DIR / "fig1_factorial_grid.png"
fig1.savefig(p1, dpi=180, bbox_inches="tight")
print(f"\n[Figure 1 saved] {p1}")

# ── Figure 2: SELL rate summary — arch x (date+group) ─────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax_i, group in enumerate(GROUPS):
    ax = axes2[ax_i]
    x  = np.arange(len(DATES))
    w  = 0.35
    sr_old = [sell_rate(CELLS[("old", group, d)]) for d in DATES]
    sr_new = [sell_rate(CELLS[("new", group, d)]) for d in DATES]
    b1 = ax.bar(x-w/2, sr_old, w, color=PALETTE["old"], alpha=0.85,
                label="Old arch (Bear last)", edgecolor="white")
    b2 = ax.bar(x+w/2, sr_new, w, color=PALETTE["new"], alpha=0.85,
                label="New arch (Judge)", edgecolor="white")
    for bar, rate in zip(list(b1)+list(b2), sr_old+sr_new):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([date_labels[d] for d in DATES], fontsize=10)
    ax.set_title(f"{group_labels[group].replace(chr(10),' ')} Ticker Group",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax.set_ylabel("SELL rate" if ax_i==0 else "")
    ax.legend(fontsize=8.5)

fig2.suptitle(f"SELL Rate by Architecture and Time Period\n"
              f"Primary test (pooled): χ²={chi2_arch:.2f}, p={p_arch:.3f}"
              + (" ***" if p_arch<0.05 else " n.s."),
              fontsize=12, fontweight="bold")
fig2.tight_layout()
p2 = FIGURES_DIR / "fig2_sell_rate_summary.png"
fig2.savefig(p2, dpi=180, bbox_inches="tight")
print(f"[Figure 2 saved] {p2}")

# ── Figure 3: BUY/SELL/HOLD stacked bar — marginal arch effect ────────────────
fig3, ax3 = plt.subplots(figsize=(6, 4.2))
arch_sigs = {"Old arch\n(Bear last)": all_old, "New arch\n(Judge)": all_new}
bottoms   = np.zeros(2)
x3        = np.arange(2)
for cat in CATS:
    vals = [sigs.count(cat)/len(sigs) for sigs in arch_sigs.values()]
    ax3.bar(x3, vals, bottom=bottoms, color=PALETTE[cat], label=cat,
            edgecolor="white", linewidth=0.8, width=0.5)
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 0.04:
            ax3.text(xi, b + v/2, f"{v:.0%}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")
    bottoms += np.array(vals)

ax3.set_xticks(x3)
ax3.set_xticklabels(list(arch_sigs.keys()), fontsize=11)
ax3.set_ylabel("Proportion", fontsize=11)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax3.set_ylim(0, 1.05)
ax3.set_title(f"Signal Composition by Architecture (n=40 each)\n"
              f"χ²={chi2_arch:.2f}  p={p_arch:.3f}"
              + (" ***" if p_arch<0.05 else " n.s.")
              + f"   |   HOLD: Fisher p={p_hold:.3f}"
              + (" ***" if p_hold<0.05 else " n.s."),
              fontsize=10.5, fontweight="bold")
legend_h = [mpatches.Patch(fc=PALETTE[c], label=c) for c in CATS]
ax3.legend(handles=legend_h, fontsize=10, loc="upper right")
fig3.tight_layout()
p3 = FIGURES_DIR / "fig3_stacked_composition.png"
fig3.savefig(p3, dpi=180, bbox_inches="tight")
print(f"[Figure 3 saved] {p3}")

plt.close("all")
print(f"\nAll figures → {FIGURES_DIR.resolve()}")
