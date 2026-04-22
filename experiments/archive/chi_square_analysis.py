"""
experiments/chi_square_analysis.py

Load results from both architecture runs and perform:
  1. Chi-square test: is signal distribution (BUY/SELL/HOLD) independent of architecture?
  2. Fisher's exact test on the 2x2 SELL vs non-SELL table (more robust for n=10).
  3. Print contingency table and key stats.
  4. Export publication-ready figures to experiments/figures/.
"""

import json
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

# ── Shared style ───────────────────────────────────────────────────────────────
PALETTE = {
    "BUY":  "#2ecc71",   # green
    "SELL": "#e74c3c",   # red
    "HOLD": "#95a5a6",   # grey
    "OLD":  "#e67e22",   # orange
    "NEW":  "#3498db",   # blue
}
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

import sys as _sys

_date_str = _sys.argv[1] if len(_sys.argv) > 1 else "2026-04-18"
OLD_PATH = f"experiments/results_old_arch_{_date_str}.json"
NEW_PATH = f"experiments/results_new_arch_{_date_str}.json"

# Normalise non-standard signals to canonical BUY/SELL/HOLD
_NORM = {
    "OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL", "STRONG BUY": "BUY",
    "STRONG SELL": "SELL", "NEUTRAL": "HOLD",
}

def _normalise(sig: str) -> str:
    return _NORM.get(sig.upper().strip(), sig.upper().strip())


def load_signals(path: str, arch_label: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    signals = []
    for row in data:
        if "error" not in row:
            signals.append(_normalise(row.get("signal", "UNKNOWN")))
        else:
            print(f"  [{arch_label}] Skipping {row.get('ticker','?')} due to error: {row['error'][:80]}")
    return signals


old_signals = load_signals(OLD_PATH, "OLD")
new_signals = load_signals(NEW_PATH, "NEW")

CATS = ["BUY", "SELL", "HOLD"]

old_counts = {c: old_signals.count(c) for c in CATS}
new_counts = {c: new_signals.count(c) for c in CATS}

# ── Contingency table ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONTINGENCY TABLE  (rows=architecture, cols=signal)")
print("="*60)
header = f"{'':12s}" + "".join(f"{c:>8s}" for c in CATS) + f"{'TOTAL':>8s}"
print(header)
print("-"*60)

old_row = [old_counts[c] for c in CATS]
new_row = [new_counts[c] for c in CATS]

print(f"{'OLD (no judge)':12s}" + "".join(f"{v:>8d}" for v in old_row) + f"{sum(old_row):>8d}")
print(f"{'NEW (judge)':12s}"    + "".join(f"{v:>8d}" for v in new_row) + f"{sum(new_row):>8d}")
print("-"*60)
col_totals = [old_counts[c]+new_counts[c] for c in CATS]
print(f"{'TOTAL':12s}" + "".join(f"{v:>8d}" for v in col_totals) + f"{sum(col_totals):>8d}")

# ── Chi-square test (2×3 table) ────────────────────────────────────────────────
observed = np.array([old_row, new_row])

# Drop columns with zero total to avoid degenerate test
valid_cols = [i for i, t in enumerate(col_totals) if t > 0]
observed_filtered = observed[:, valid_cols]
labels_filtered   = [CATS[i] for i in valid_cols]

chi2, p_chi2, dof, expected = stats.chi2_contingency(observed_filtered, correction=False)

print("\n" + "="*60)
print("CHI-SQUARE TEST  (2×K table, K = non-zero signal categories)")
print("="*60)
print(f"  Categories used : {labels_filtered}")
print(f"  Chi2 statistic  : {chi2:.4f}")
print(f"  Degrees of freedom: {dof}")
print(f"  p-value         : {p_chi2:.4f}")
if p_chi2 < 0.05:
    print("  → SIGNIFICANT at α=0.05: signal distribution differs between architectures.")
else:
    print("  → NOT significant at α=0.05 (but n=10 per group limits power).")

# ── Fisher's exact: SELL vs non-SELL (2×2) ────────────────────────────────────
old_sell     = old_counts["SELL"]
old_non_sell = len(old_signals) - old_sell
new_sell     = new_counts["SELL"]
new_non_sell = len(new_signals) - new_sell

table_2x2 = [[old_sell, old_non_sell],
             [new_sell, new_non_sell]]

odds_ratio, p_fisher = stats.fisher_exact(table_2x2, alternative="two-sided")

print("\n" + "="*60)
print("FISHER'S EXACT TEST  (SELL vs non-SELL, 2×2)")
print("="*60)
print(f"  OLD arch: SELL={old_sell}, non-SELL={old_non_sell}")
print(f"  NEW arch: SELL={new_sell}, non-SELL={new_non_sell}")
print(f"  Odds ratio : {odds_ratio:.4f}")
print(f"  p-value    : {p_fisher:.4f}")
if p_fisher < 0.05:
    print("  → SIGNIFICANT: SELL proportion differs between architectures.")
else:
    print("  → NOT significant at α=0.05.")

# ── SELL proportion comparison ─────────────────────────────────────────────────
n_old = len(old_signals)
n_new = len(new_signals)
p_old_sell = old_sell / n_old if n_old else 0
p_new_sell = new_sell / n_new if n_new else 0

print("\n" + "="*60)
print("SELL PROPORTION SUMMARY")
print("="*60)
print(f"  OLD arch SELL rate: {old_sell}/{n_old} = {p_old_sell:.1%}")
print(f"  NEW arch SELL rate: {new_sell}/{n_new} = {p_new_sell:.1%}")
print(f"  Difference (OLD−NEW): {p_old_sell - p_new_sell:+.1%}")

# ── Per-ticker breakdown ───────────────────────────────────────────────────────
def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return {r["ticker"]: r for r in json.load(f) if "error" not in r}

old_rows = load_rows(OLD_PATH)
new_rows = load_rows(NEW_PATH)

tickers = sorted(set(old_rows) | set(new_rows))

print("\n" + "="*60)
print("PER-TICKER COMPARISON")
print("="*60)
print(f"{'Ticker':8s}  {'OLD signal':12s}  {'NEW signal':12s}  {'Match?':8s}")
print("-"*60)
matches = 0
total   = 0
for t in tickers:
    o = _normalise(old_rows.get(t, {}).get("signal", "N/A"))
    n = _normalise(new_rows.get(t, {}).get("signal", "N/A"))
    match = "✓" if o == n else "✗"
    if o != "N/A" and n != "N/A":
        total += 1
        if o == n:
            matches += 1
    print(f"{t:8s}  {o:12s}  {n:12s}  {match}")

if total:
    print(f"\n  Agreement rate: {matches}/{total} = {matches/total:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Grouped bar chart — signal distribution by architecture ──────────
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

x      = np.arange(len(CATS))
width  = 0.35
bars_old = ax1.bar(x - width/2, old_row, width,
                   color=[PALETTE[c] for c in CATS], alpha=0.75,
                   edgecolor="white", linewidth=0.8, label="Old arch (no Judge)")
bars_new = ax1.bar(x + width/2, new_row, width,
                   color=[PALETTE[c] for c in CATS], alpha=1.0,
                   edgecolor="white", linewidth=0.8, label="New arch (Judge-mediated)")

# Hatch old bars to distinguish from new
for bar in bars_old:
    bar.set_hatch("///")

# Value labels
for bar in list(bars_old) + list(bars_new):
    h = bar.get_height()
    if h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(CATS, fontsize=12)
ax1.set_ylabel("Number of tickers", fontsize=11)
ax1.set_title("Signal Distribution: Old vs New Debate Architecture\n"
              f"(n=10 each, date=2026-04-18)",
              fontsize=12, fontweight="bold", pad=12)
ax1.set_ylim(0, max(max(old_row), max(new_row)) + 1.8)
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Custom legend: arch style + color meaning
legend_handles = [
    mpatches.Patch(facecolor="#aaa", hatch="///", edgecolor="white", label="Old arch (Bear last)"),
    mpatches.Patch(facecolor="#aaa", edgecolor="white",              label="New arch (Judge)"),
    mpatches.Patch(facecolor=PALETTE["BUY"],  label="BUY"),
    mpatches.Patch(facecolor=PALETTE["SELL"], label="SELL"),
    mpatches.Patch(facecolor=PALETTE["HOLD"], label="HOLD"),
]
ax1.legend(handles=legend_handles, fontsize=9, loc="upper right",
           framealpha=0.85, edgecolor="#ccc")

# Annotate p-value
sig_text = (f"Chi² = {chi2:.3f}, p = {p_chi2:.3f}"
            + (" *" if p_chi2 < 0.05 else " (n.s.)"))
ax1.text(0.02, 0.97, sig_text, transform=ax1.transAxes,
         fontsize=9, va="top", color="#555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ccc"))

fig1.tight_layout()
path1 = FIGURES_DIR / "fig1_signal_distribution.png"
fig1.savefig(path1, dpi=180, bbox_inches="tight")
print(f"\n[Figure 1 saved] {path1}")

# ── Figure 2: SELL proportion comparison with Fisher p-value ──────────────────
fig2, ax2 = plt.subplots(figsize=(5, 4))

arch_labels = ["Old arch\n(Bear last)", "New arch\n(Judge)"]
sell_rates  = [p_old_sell, p_new_sell]
bar_colors  = [PALETTE["OLD"], PALETTE["NEW"]]

bars2 = ax2.bar(arch_labels, sell_rates, color=bar_colors,
                width=0.45, edgecolor="white", linewidth=0.8)

for bar, rate in zip(bars2, sell_rates):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.012,
             f"{rate:.0%}",
             ha="center", va="bottom", fontsize=13, fontweight="bold")

ax2.set_ylabel("SELL signal rate", fontsize=11)
ax2.set_ylim(0, max(sell_rates) + 0.20)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.set_title("SELL Rate by Architecture\n"
              f"(Fisher's exact: OR={odds_ratio:.2f}, p={p_fisher:.3f}"
              + (" *)" if p_fisher < 0.05 else " n.s.)"),
              fontsize=11, fontweight="bold", pad=10)

# Significance bracket if significant
if p_fisher < 0.05:
    y_top = max(sell_rates) + 0.10
    ax2.plot([0, 0, 1, 1], [y_top-0.03, y_top, y_top, y_top-0.03],
             color="#333", linewidth=1.2)
    sig_label = "***" if p_fisher < 0.001 else ("**" if p_fisher < 0.01 else "*")
    ax2.text(0.5, y_top + 0.01, sig_label, ha="center", fontsize=14, color="#333")

fig2.tight_layout()
path2 = FIGURES_DIR / "fig2_sell_rate.png"
fig2.savefig(path2, dpi=180, bbox_inches="tight")
print(f"[Figure 2 saved] {path2}")

# ── Figure 3: Per-ticker heatmap (OLD vs NEW) ─────────────────────────────────
SIG_NUM = {"BUY": 1, "HOLD": 0, "SELL": -1, "N/A": np.nan, "UNKNOWN": np.nan}
SIG_COLOR = {"BUY": PALETTE["BUY"], "SELL": PALETTE["SELL"],
             "HOLD": PALETTE["HOLD"], "N/A": "#eeeeee", "UNKNOWN": "#eeeeee"}

all_tickers = sorted(set(old_rows) | set(new_rows))
n = len(all_tickers)

fig3, ax3 = plt.subplots(figsize=(5, max(3.5, n * 0.55 + 1.2)))

cell_w, cell_h = 1.0, 0.8
for row_i, ticker in enumerate(all_tickers):
    o_sig = old_rows.get(ticker, {}).get("signal", "N/A")
    n_sig = new_rows.get(ticker, {}).get("signal", "N/A")

    for col_i, (sig, arch) in enumerate([(o_sig, "Old"), (n_sig, "New")]):
        rect = plt.Rectangle((col_i * cell_w, row_i * cell_h),
                              cell_w, cell_h,
                              facecolor=SIG_COLOR.get(sig, "#eeeeee"),
                              edgecolor="white", linewidth=1.5)
        ax3.add_patch(rect)
        ax3.text(col_i * cell_w + cell_w/2,
                 row_i * cell_h + cell_h/2,
                 sig, ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white")

# Axis labels
ax3.set_xlim(0, 2 * cell_w)
ax3.set_ylim(0, n * cell_h)
ax3.set_xticks([cell_w/2, 1.5*cell_w])
ax3.set_xticklabels(["Old arch\n(Bear last)", "New arch\n(Judge)"], fontsize=10)
ax3.set_yticks([i * cell_h + cell_h/2 for i in range(n)])
ax3.set_yticklabels(all_tickers, fontsize=10)
ax3.tick_params(left=False, bottom=False)
for spine in ax3.spines.values():
    spine.set_visible(False)
ax3.set_title("Per-ticker Signal Comparison", fontsize=12, fontweight="bold", pad=10)

fig3.tight_layout()
path3 = FIGURES_DIR / "fig3_per_ticker_heatmap.png"
fig3.savefig(path3, dpi=180, bbox_inches="tight")
print(f"[Figure 3 saved] {path3}")

plt.close("all")
print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")
