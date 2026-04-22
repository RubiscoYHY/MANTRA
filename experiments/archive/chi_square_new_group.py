"""
experiments/chi_square_new_group.py

Combined chi-square analysis for the NEW ticker group
(AVGO, TSM, BRK.B, WMT, LLY, V, MU, ORCL, MA, AMD)
across all available dates.

OLD arch files: results_old_arch_new_*.json
NEW arch files: results_new_arch_new_*.json
"""

import json
import glob
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
           "OLD": "#e67e22", "NEW": "#3498db"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

_NORM = {"OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL", "STRONG BUY": "BUY",
         "STRONG SELL": "SELL", "NEUTRAL": "HOLD"}
def _n(s): return _NORM.get(s.upper().strip(), s.upper().strip())

# ── Load ───────────────────────────────────────────────────────────────────────
def load(pattern, label):
    files = sorted(glob.glob(pattern))
    print(f"  {label}: {[Path(f).name for f in files]}")
    rows, sigs = [], []
    for path in files:
        date_tag = Path(path).stem.split("_")[-1]
        for row in json.load(open(path, encoding="utf-8")):
            if "error" not in row:
                sig = _n(row.get("signal", "UNKNOWN"))
                sigs.append(sig)
                rows.append({**row, "signal": sig, "date": date_tag})
    return rows, sigs

print("Loading result files:")
old_rows, old_sigs = load("experiments/results_old_arch_new_*.json", "OLD")
new_rows, new_sigs = load("experiments/results_new_arch_new_*.json", "NEW")

CATS   = ["BUY", "SELL", "HOLD"]
n_old, n_new = len(old_sigs), len(new_sigs)
old_c  = {c: old_sigs.count(c) for c in CATS}
new_c  = {c: new_sigs.count(c) for c in CATS}
old_row = [old_c[c] for c in CATS]
new_row = [new_c[c] for c in CATS]
col_tot = [old_c[c]+new_c[c] for c in CATS]

# ── Print table ────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"CONTINGENCY TABLE — New ticker group  (n_old={n_old}, n_new={n_new})")
print(f"{'='*65}")
print(f"{'':14s}" + "".join(f"{c:>8s}" for c in CATS) + f"{'TOTAL':>8s}")
print("-"*65)
print(f"{'OLD (no judge)':14s}" + "".join(f"{v:>8d}" for v in old_row) + f"{n_old:>8d}")
print(f"{'NEW (judge)':14s}"    + "".join(f"{v:>8d}" for v in new_row) + f"{n_new:>8d}")
print("-"*65)
print(f"{'TOTAL':14s}"          + "".join(f"{v:>8d}" for v in col_tot) + f"{n_old+n_new:>8d}")

# ── Chi-square ─────────────────────────────────────────────────────────────────
obs = np.array([old_row, new_row])
vcols = [i for i, t in enumerate(col_tot) if t > 0]
chi2, p_chi2, dof, exp = stats.chi2_contingency(obs[:, vcols], correction=False)
print(f"\n{'='*65}")
print("CHI-SQUARE TEST")
print(f"{'='*65}")
print(f"  Categories : {[CATS[i] for i in vcols]}")
print(f"  Chi²       : {chi2:.4f}   df={dof}   p={p_chi2:.4f}  "
      f"{'*** SIGNIFICANT' if p_chi2 < 0.05 else '(not significant)'}")

# ── Fisher SELL vs non-SELL ────────────────────────────────────────────────────
os_, ons = old_c["SELL"], n_old - old_c["SELL"]
ns_, nns = new_c["SELL"], n_new - new_c["SELL"]
OR, p_f  = stats.fisher_exact([[os_, ons], [ns_, nns]], alternative="two-sided")
p_os, p_ns = os_/n_old, ns_/n_new

print(f"\n{'='*65}")
print("FISHER'S EXACT TEST  (SELL vs non-SELL)")
print(f"{'='*65}")
print(f"  OLD  SELL={os_}/{n_old} ({p_os:.0%})   non-SELL={ons}")
print(f"  NEW  SELL={ns_}/{n_new} ({p_ns:.0%})   non-SELL={nns}")
print(f"  OR={OR:.3f}   p={p_f:.4f}  "
      f"{'*** SIGNIFICANT' if p_f < 0.05 else '(not significant)'}")
print(f"  SELL diff (OLD−NEW): {p_os-p_ns:+.1%}")

# ── Per-date breakdown ─────────────────────────────────────────────────────────
dates = sorted({r["date"] for r in old_rows+new_rows})
print(f"\n{'='*65}\nPER-DATE BREAKDOWN\n{'='*65}")
for d in dates:
    os = [r["signal"] for r in old_rows if r["date"]==d]
    ns = [r["signal"] for r in new_rows if r["date"]==d]
    ns_str = f"BUY={ns.count('BUY')} SELL={ns.count('SELL')} HOLD={ns.count('HOLD')}" if ns else "— not run —"
    print(f"  {d}:")
    print(f"    OLD  BUY={os.count('BUY')} SELL={os.count('SELL')} HOLD={os.count('HOLD')}  (SELL {os.count('SELL')/len(os):.0%})")
    print(f"    NEW  {ns_str}" + (f"  (SELL {ns.count('SELL')/len(ns):.0%})" if ns else ""))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
n_dates = len(dates)
fig, axes = plt.subplots(1, n_dates+1,
                         figsize=(4*(n_dates+1), 4.8),
                         gridspec_kw={"width_ratios": [1]*n_dates + [1.3]})
if n_dates == 1:
    axes = [axes, plt.subplot(1,1,1)]   # fallback — shouldn't happen

for ax_i, d in enumerate(dates):
    ax   = axes[ax_i]
    os   = [r["signal"] for r in old_rows if r["date"]==d]
    ns   = [r["signal"] for r in new_rows if r["date"]==d]
    oc   = [os.count(c) for c in CATS]
    nc   = [ns.count(c) for c in CATS] if ns else [0]*3
    x, w = np.arange(3), 0.35
    bo   = ax.bar(x-w/2, oc, w, color=[PALETTE[c] for c in CATS], alpha=0.72,
                  edgecolor="white", hatch="///")
    bn   = ax.bar(x+w/2, nc, w, color=[PALETTE[c] for c in CATS], alpha=1.0,
                  edgecolor="white")
    for bar in list(bo)+list(bn):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.06, str(int(h)),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=10)
    ax.set_title(f"{d}", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(max(oc), max(nc) if nc else 0) + 2.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel("Count" if ax_i == 0 else "")

# Combined panel
ax_c = axes[-1]
x, w = np.arange(3), 0.35
bo = ax_c.bar(x-w/2, old_row, w, color=[PALETTE[c] for c in CATS], alpha=0.72,
              edgecolor="white", hatch="///")
bn = ax_c.bar(x+w/2, new_row, w, color=[PALETTE[c] for c in CATS], alpha=1.0,
              edgecolor="white")
for bar in list(bo)+list(bn):
    h = bar.get_height()
    if h > 0:
        ax_c.text(bar.get_x()+bar.get_width()/2, h+0.06, str(int(h)),
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_c.set_xticks(x); ax_c.set_xticklabels(CATS, fontsize=10)
ax_c.set_title("Combined", fontsize=10, fontweight="bold")
ax_c.set_ylim(0, max(max(old_row), max(new_row)) + 2.5)
ax_c.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
sig_txt = f"χ²={chi2:.2f}, p={p_chi2:.3f}" + (" *" if p_chi2<0.05 else " n.s.")
ax_c.text(0.03, 0.97, sig_txt, transform=ax_c.transAxes, fontsize=8, va="top",
          color="#555", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ccc"))

legend_h = [
    mpatches.Patch(fc="#aaa", hatch="///", ec="white", label="Old arch (Bear last)"),
    mpatches.Patch(fc="#aaa", ec="white",              label="New arch (Judge)"),
    mpatches.Patch(fc=PALETTE["BUY"],  label="BUY"),
    mpatches.Patch(fc=PALETTE["SELL"], label="SELL"),
    mpatches.Patch(fc=PALETTE["HOLD"], label="HOLD"),
]
fig.legend(handles=legend_h, fontsize=9, loc="lower center", ncol=5,
           framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Signal Distribution — New Ticker Group\n"
             "(AVGO, TSM, BRK.B, WMT, LLY, V, MU, ORCL, MA, AMD)",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
p1 = FIGURES_DIR / "figC_new_group_distribution.png"
fig.savefig(p1, dpi=180, bbox_inches="tight")
print(f"\n[Figure C saved] {p1}")

# ── SELL rate bar ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5 + n_dates*0.8, 4.2))
x_labels   = dates + ["Combined"]
sell_old_r = [sum(1 for r in old_rows if r["date"]==d and r["signal"]=="SELL") /
              max(sum(1 for r in old_rows if r["date"]==d), 1) for d in dates] + [p_os]
sell_new_r = [sum(1 for r in new_rows if r["date"]==d and r["signal"]=="SELL") /
              max(sum(1 for r in new_rows if r["date"]==d), 1) for d in dates] + [p_ns]

x, w = np.arange(len(x_labels)), 0.35
b1 = ax2.bar(x-w/2, sell_old_r, w, color=PALETTE["OLD"], alpha=0.85,
             label="Old arch (Bear last)", edgecolor="white")
b2 = ax2.bar(x+w/2, sell_new_r, w, color=PALETTE["NEW"], alpha=0.85,
             label="New arch (Judge)", edgecolor="white")
for bar, rate in zip(list(b1)+list(b2), sell_old_r+sell_new_r):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
             f"{rate:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(x_labels, fontsize=11)
ax2.set_ylabel("SELL rate", fontsize=11)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax2.set_ylim(0, max(sell_old_r+sell_new_r)+0.22)
ax2.set_title(f"SELL Rate by Date — New Ticker Group\n"
              f"Fisher (combined): OR={OR:.2f}, p={p_f:.3f}"
              + (" *" if p_f<0.05 else " (n.s.)"), fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.axvline(n_dates-0.5, color="#bbb", linestyle="--", linewidth=1)
fig2.tight_layout()
p2 = FIGURES_DIR / "figD_new_group_sell_rate.png"
fig2.savefig(p2, dpi=180, bbox_inches="tight")
print(f"[Figure D saved] {p2}")

plt.close("all")
print(f"\nAll figures → {FIGURES_DIR.resolve()}")
