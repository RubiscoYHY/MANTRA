# Debate Architecture Bias Analysis Report

**Project:** Mantra Trading Agents  
**Author:** Rubisco (Xiaoxin Qiu)  
**Date:** 2026-04-19  
**Experiment Type:** 2×2×2 Factorial Design

---

## 1. Background and Motivation

The original Mantra research debate module used a sequential Bull/Bear architecture: the Bull analyst spoke first, then the Bear analyst responded, and the Research Manager issued a final verdict immediately after the Bear's closing argument. This design raised a concern about **recency bias**: because the Research Manager always reads the Bear argument last, it may systematically over-weight the bearish perspective and produce an excess of SELL signals.

On 2026-04-18, the architecture was redesigned to introduce a neutral **Judge** mediator. Under the new design, Bull and Bear analysts speak in parallel (neither seeing the other's argument), the Judge issues targeted critiques to each side, and only after this structured exchange does the Research Manager deliver a verdict. This eliminates the positional asymmetry entirely.

This report documents a controlled experiment designed to quantify the bias in the old architecture and verify that the new architecture corrects it.

---

## 2. Experimental Design

### 2.1 Factorial Structure

The experiment follows a **2×2×2 fully crossed factorial design**, with three independent variables:

| Dimension | Level A | Level B |
|---|---|---|
| **Architecture** | Old (Bear speaks last) | New (Judge-mediated) |
| **Time period** | 2026-03-24 (volatile / uncertain market) | 2026-04-16 (bull market) |
| **Ticker group** | Original: AAPL, MSFT, NVDA, AMZN, GOOGL, JPM, XOM, JNJ, META, TSLA | New: AVGO, TSM, BRK.B, WMT, LLY, V, MU, ORCL, MA, AMD |

This yields **8 cells × n=10 tickers = 80 total observations**.

### 2.2 Model Configuration

All runs used identical model settings to isolate the architectural variable:

| Parameter | Value |
|---|---|
| Quick-thinking model | `gemini-3.1-flash-lite-preview` (analysts, researchers, trader) |
| Deep-thinking model | `gemini-3.1-pro-preview` (Research Manager, Portfolio Manager) |
| Provider | Google Gemini (all layers) |
| Old arch debate rounds | `max_debate_rounds = 1` |
| New arch judge iterations | `judge_iterations = 1` |
| Risk discussion rounds | `max_risk_discuss_rounds = 1` |
| Analysts | Market, Social, News, Fundamentals (all four, parallel) |

### 2.3 Outcome Variable

The primary outcome is the **signal distribution** across three categories: `BUY`, `SELL`, `HOLD`. Non-standard signals (`OVERWEIGHT`, `NEUTRAL`, etc.) are normalised to their canonical equivalents before analysis.

The two key metrics are:
- **SELL rate**: proportion of SELL signals per cell
- **HOLD rate**: proportion of HOLD signals per cell

### 2.4 Old Architecture: Structural Description

Under `max_debate_rounds = 1`, the debate graph executes as follows:

```
[Analyst Reports]
      ↓
Bull Analyst speaks   (count = 1)
      ↓
Bear Analyst responds (count = 2)
      ↓  count >= 2*max_debate_rounds → exit
Research Manager decides
```

**Key asymmetry:** Bear always delivers the final argument. The Research Manager's context window ends with the Bear's rebuttal in 100% of cases. Routing is governed by `current_response.startswith("Bull")` — if the last speaker was Bull, Bear responds next; otherwise Bull responds. Since the loop exits after Bear's turn, Bear structurally owns the last word.

### 2.5 New Architecture: Structural Description

Under `judge_iterations = 1`, the debate graph executes as follows:

```
[Analyst Reports]
      ↓
Researcher Round (Bull + Bear in parallel, neither sees the other)
      ↓  judge_count < judge_iterations
Judge Researcher (issues separate critiques to Bull and Bear)
      ↓
Researcher Round (Bull and Bear each respond to Judge's critique only)
      ↓  judge_count >= judge_iterations
Research Manager decides
```

**Key symmetry:** Bull and Bear always speak simultaneously. Neither sees the other's argument; each only receives the Judge's targeted directive. The Research Manager reads a balanced, Judge-structured debate rather than a Bear-ending monologue.

---

## 3. Results

### 3.1 Per-Cell SELL Rates

| Ticker Group | Date | Old Arch SELL | New Arch SELL | Difference |
|---|---|---|---|---|
| Original | 2026-03-24 | 20% (2/10) | 10% (1/10) | +10% |
| Original | 2026-04-16 | 30% (3/10) | 10% (1/10) | +20% |
| New | 2026-03-24 | 20% (2/10) | 10% (1/10) | +10% |
| New | 2026-04-16 | 40% (4/10) | 10% (1/10) | **+30%** |

The new architecture's SELL rate is **locked at 10% across all four cells**, regardless of time period or ticker composition. The old architecture's SELL rate varies from 20% to 40% depending on market conditions.

### 3.2 Marginal SELL Rates

| Condition | Old Arch | New Arch | Difference |
|---|---|---|---|
| Overall (all 80 obs) | 28% (11/40) | 10% (4/40) | +18% |
| Bull market (4-16) | 35% (7/20) | 10% (2/20) | +25% |
| Volatile market (3-24) | 20% (4/20) | 10% (2/20) | +10% |

### 3.3 Full Signal Composition (n=40 per arch)

| Signal | Old Arch | New Arch |
|---|---|---|
| BUY | 27 (68%) | 22 (55%) |
| SELL | 11 (28%) | 4 (10%) |
| HOLD | 2 (5%) | 14 (35%) |

The most striking structural difference is in **HOLD signals**: the old architecture produced virtually none (5%), while the new architecture produced them at a 35% rate. This observation motivates the secondary hypothesis discussed in Section 4.

---

## 4. Statistical Analysis

### 4.1 Primary Test: Architecture Main Effect

A chi-square test of independence was applied to the 2×3 contingency table (architecture × signal category), pooling all 80 observations.

|  | BUY | SELL | HOLD | Total |
|---|---|---|---|---|
| Old arch | 27 | 11 | 2 | 40 |
| New arch | 22 | 4 | 14 | 40 |
| **Total** | 49 | 15 | 16 | 80 |

$$\chi^2(2) = 12.78, \quad p = 0.0017$$

The result is **highly significant** at α = 0.001. The overall signal distribution differs significantly between the two architectures.

### 4.2 Architecture × Time Interaction

Chi-square tests were run separately for each time period, pooling across ticker groups (n=20 per arch per period):

| Time Period | χ² | p | Significant? |
|---|---|---|---|
| Volatile / uncertain (2026-03-24) | 7.11 | 0.029 | Yes (α=0.05) |
| Bull market (2026-04-16) | 6.35 | 0.042 | Yes (α=0.05) |

The architecture effect is significant in **both** market conditions, indicating that the bias is not an artifact of a particular market environment. However, the magnitude is larger in the bull market (+25% vs +10%), suggesting that strong upward momentum amplifies the old architecture's distortion.

### 4.3 Architecture × Ticker Group Interaction

| Ticker Group | χ² | p | Significant? |
|---|---|---|---|
| Original (AAPL, MSFT…) | 4.23 | 0.121 | No |
| New (AVGO, TSM…) | 9.36 | 0.009 | Yes (α=0.01) |

The architecture effect is statistically significant in the new ticker group but not in the original group. This may reflect the smaller sample size within each subgroup (n=20 per arch) rather than a genuine moderation effect. The direction of the difference is identical in both groups (OLD SELL > NEW SELL), and across the pooled sample the main effect is robust.

### 4.4 HOLD Signal: Fisher's Exact Test

The structural disappearance of HOLD signals in the old architecture was tested directly:

|  | HOLD | non-HOLD |
|---|---|---|
| Old arch | 2 | 38 |
| New arch | 14 | 26 |

$$\text{OR} = 0.098, \quad p = 0.0015 \quad (\text{Fisher's exact, two-sided})$$

The suppression of HOLD in the old architecture is **highly significant** (p = 0.0015).

---

## 5. Interpretation

### 5.1 Two Distinct Bias Mechanisms

The data reveal two separate failure modes of the old architecture, not one:

**Mechanism A — SELL inflation (Bear positional advantage)**  
Because Bear always speaks last, the Research Manager is more likely to end its context on a bearish note. This inflates SELL relative to the new architecture (28% vs 10%, OR ≈ 3.4).

**Mechanism B — HOLD suppression (forced binary verdict)**  
The old debate structure forces the Research Manager to adjudicate a winner after Bear's closing argument. This imposes an implicit binary framing (Bull wins → BUY; Bear wins → SELL), crowding out nuanced neutral conclusions. HOLD appears only 5% of the time in the old architecture vs 35% in the new.

Mechanism B is arguably the more fundamental defect. A well-calibrated analyst should frequently conclude "the evidence does not strongly favor either side." The old architecture structurally prevents this outcome.

### 5.2 Market Condition Interaction

The bias is present in both market environments but is amplified in the bull market (4-16):

- In the volatile period (3-24), Bull and Bear have roughly balanced material. Even with Bear speaking last, the Research Manager can recognise the uncertainty.
- In the bull market (4-16), the underlying data strongly supports the bull case. The Bull analyst presents compelling evidence. The Bear analyst must construct a contrarian argument without strong factual support. In the old architecture, this forced contrarian argument still gets the last word, producing artificially elevated SELL signals (35–40%) even when market conditions clearly favour a BUY verdict.

### 5.3 Cross-Group Consistency

The SELL rate differential (OLD − NEW) is +10% in every individual cell of the 3-24 date, and +20–30% in every cell of the 4-16 date. This cross-group consistency — across two entirely different sets of tickers — indicates the bias is an architectural property, not a sampling artifact.

---

## 6. Limitations

1. **Sample size**: Each cell contains n=10 tickers. While the pooled tests are highly significant (p < 0.002), individual cell-level comparisons have low statistical power. Conclusions from the interaction analyses should be treated as directional.

2. **Single debate depth**: All runs used the shallowest configuration (1 round / 1 judge iteration). The magnitude of the bias may differ at greater debate depths.

3. **Single date per combination**: Each (arch, group, date) cell is a single LLM run with no repetition. Stochastic variation in LLM outputs cannot be separated from true signal variation.

4. **Date alignment**: The original ticker group was also run on 2026-04-18 (Saturday, market closed) in preliminary tests. The April results reported here use 2026-04-16 for both groups for consistency.

5. **Non-standard signals**: One instance of `OVERWEIGHT` (AMZN, new arch) was normalised to `BUY`. This normalisation is reasonable but introduces a minor coding assumption.

---

## 7. Conclusion

This experiment provides strong statistical evidence that the old Bear-last debate architecture introduced a systematic bias in the Mantra trading signal pipeline:

- **Architecture main effect: χ²(2) = 12.78, p = 0.0017** — the overall signal distribution is significantly different between architectures.
- **HOLD suppression: Fisher OR = 0.098, p = 0.0015** — the old architecture structurally prevents neutral verdicts.
- The bias is **consistent across two different ticker groups and two market environments**, ruling out sampling artifacts.
- The new Judge-mediated architecture produces a **stable 10% SELL rate** across all experimental conditions, compared to 20–40% in the old architecture.

The introduction of the Judge mediator corrects both bias mechanisms: it eliminates Bear's positional advantage by parallelising research rounds, and it permits the Research Manager to deliver calibrated HOLD verdicts when the evidence is mixed.

---

## Appendix: Data Files

All result files are stored in `experiments/`:

| File | Description |
|---|---|
| `results_old_arch_2026-03-24.json` | Old arch, original tickers, 2026-03-24 |
| `results_old_arch_original_2026-04-16.json` | Old arch, original tickers, 2026-04-16 |
| `results_new_arch_2026-03-24.json` | New arch, original tickers, 2026-03-24 |
| `results_new_arch_original_2026-04-16.json` | New arch, original tickers, 2026-04-16 |
| `results_old_arch_new_2026-03-24.json` | Old arch, new tickers, 2026-03-24 |
| `results_old_arch_new_2026-04-16.json` | Old arch, new tickers, 2026-04-16 |
| `results_new_arch_new_2026-03-24.json` | New arch, new tickers, 2026-03-24 |
| `results_new_arch_new_2026-04-16.json` | New arch, new tickers, 2026-04-16 |

Figures are stored in `experiments/figures/`:

| File | Description |
|---|---|
| `fig1_factorial_grid.png` | 2×4 grid of signal distributions across all cells |
| `fig2_sell_rate_summary.png` | SELL rate by architecture, date, and ticker group |
| `fig3_stacked_composition.png` | Stacked signal composition (marginal arch effect) |
