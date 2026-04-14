# MANTRA

**Memory-Augmented Neural Trading Retrieval Agents**

A multi-agent LLM framework for equity analysis, built on [TradingAgents](https://github.com/TauricResearch/TradingAgents) with a persistent memory system, FinBERT-based sentiment preprocessing, and a backtesting engine.

---

## Installation

```bash
git clone https://github.com/RubiscoYHY/MANTRA.git
cd MANTRA
conda create -n mantra python=3.13
conda activate mantra
pip install -e .
```

To use FinBERT sentiment preprocessing (recommended):

```bash
pip install transformers torch
```

FinBERT weights (~440 MB) are downloaded automatically on first run and cached at `~/.cache/huggingface/`. If you skip this step, the Social Media Analyst will fall back to passing raw posts directly to the LLM.

**GPU acceleration (CUDA / Apple Silicon):** MANTRA automatically detects the best available device at runtime — `cuda` on NVIDIA GPUs, `mps` on Apple M-series, `cpu` otherwise. FinBERT inference and BGE embeddings both use this device without any manual configuration.

> **Windows (NVIDIA GPU):** `pip install torch` fetches the CPU-only wheel by default on Windows. To enable CUDA, install the GPU-enabled build from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and select your CUDA version. Example for CUDA 12.1:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install transformers
> ```
> Everything else in the setup is identical — `cp` is a valid alias in PowerShell and works as shown above.

### Environment Variables

Copy `.env.example` to `.env` and fill in the keys for your chosen providers:

```bash
cp .env.example .env
```

```bash
ANTHROPIC_API_KEY=sk-ant-...    # Claude (recommended for manager layer)
GOOGLE_API_KEY=AIza...          # Gemini
OPENAI_API_KEY=sk-...           # OpenAI
HF_TOKEN=hf_...                 # HuggingFace Inference API
XAI_API_KEY=...                 # xAI (Grok)
OPENROUTER_API_KEY=...          # OpenRouter
ALPHA_VANTAGE_API_KEY=...       # Market data
```

You only need keys for the providers you actually use. For fully local inference, Ollama requires no API key.

### Supported LLM Providers

MANTRA uses a dual-LLM architecture: a **deep-thinking** model for the manager layer (Research Manager, Portfolio Manager) and a **quick-thinking** model for everything else (analysts, researchers, trader, risk debate).

| Provider | `provider` value | Key |
|----------|-----------------|-----|
| OpenAI | `"openai"` | `OPENAI_API_KEY` |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` |
| Google | `"google"` | `GOOGLE_API_KEY` |
| xAI | `"xai"` | `XAI_API_KEY` |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` |
| HuggingFace | `"huggingface"` | `HF_TOKEN` |
| Ollama (local) | `"ollama"` | none |

---

## CLI

```bash
mantra
```

The interactive CLI walks through seven steps: ticker, analysis date, output language, analyst team selection, research depth, quick-think LLM, and deep-think LLM. Provider and model are selected independently at each step.

To run a backtest:

```bash
mantra --mode backtest
```

The backtest CLI prompts for ticker, date range, and model configuration, then runs day-by-day analysis. Results are saved automatically to `results/{TICKER}-{start}-{end}/`.

---

## Python API

### Single-day analysis

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

load_dotenv()

config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "anthropic"
config["deep_think_llm"]       = "claude-opus-4-6"
config["quick_think_provider"] = "google"
config["quick_think_llm"]      = "gemini-2.5-flash"
# Analysts run in parallel by default for cloud providers.
# Set to False to force sequential execution, or True to enable parallel even for local models.
# config["parallel_analysts"] = False

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

`propagate()` returns a tuple `(state, decision)`. The second element is a dictionary:

```python
{
    "signal":     "BUY",    # BUY | OVERWEIGHT | HOLD | UNDERWEIGHT | SELL
    "confidence": 0.82,     # float in [0, 1]
    "horizon":    "short"   # investment horizon
}
```

The first element `state` is the full LangGraph state dict containing all intermediate agent reports, which you can inspect for debugging.

### Google Colab / Jupyter

**Setup (run once at the top of your notebook):**

```python
# Clone and install
!git clone https://github.com/RubiscoYHY/MANTRA.git
%cd MANTRA
!pip install -e . -q

# Install transformers — torch is already CUDA-enabled in Colab, so skip reinstalling it
!pip install transformers -q
```

Colab runtimes (T4, A100, L4) ship with a CUDA-enabled PyTorch pre-installed. MANTRA's auto-detection picks this up automatically — FinBERT inference and BGE embeddings will run on the GPU without any extra configuration.

**API keys and usage:**

Set keys directly via `os.environ` before initializing the graph (`.env` files are not available in Colab). Set `debug=False` to suppress the live-streaming output:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["GOOGLE_API_KEY"] = "AIza..."

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "anthropic"
config["deep_think_llm"]       = "claude-opus-4-6"
config["quick_think_provider"] = "google"
config["quick_think_llm"]      = "gemini-2.5-flash"

ta = TradingAgentsGraph(debug=False, config=config)
_, decision = ta.propagate("AAPL", "2024-03-15")
print(decision["signal"], decision["confidence"])
```

> **Note:** If you restart the Colab runtime, re-run the `%cd MANTRA` cell before importing — Python's working directory resets on restart.

### Backtesting

Use `run_backtest()` to run analysis across a date range. The memory system accumulates across days; each day's reflections become available to subsequent days.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "anthropic"
config["deep_think_llm"]       = "claude-opus-4-6"
config["quick_think_provider"] = "google"
config["quick_think_llm"]      = "gemini-2.5-flash"
config["run_mode"]             = "backtest"

ta = TradingAgentsGraph(debug=False, config=config)
results = ta.run_backtest("NVDA", "2024-01-02", "2024-03-29")
```

`run_backtest()` returns a DataFrame with columns `date`, `signal`, `confidence`, and `actual_return`. It also writes the following to `results/{TICKER}-{start}-{end}/`:

- `backtest_{TICKER}_{timestamp}.csv` — the full day-by-day results
- `analysis.png` — four-panel chart: equity curves, position stacking, confidence distribution, and calibration plot
- `metrics.csv` — Sharpe ratio, cumulative return, max drawdown, and win rate for each strategy (TA-Signal, TA-Filtered, TA-Scaled) against five traditional baselines

To regenerate charts from an existing CSV:

```python
import pandas as pd
from tradingagents.graph.backtest_analyze import _build_figure, _download_ohlcv, _write_metrics_csv

df = pd.read_csv("results/NVDA-2024-01-02-2024-03-29/backtest_NVDA_....csv")
prices = _download_ohlcv("NVDA", "2024-01-02", "2024-04-08")
fig, values = _build_figure(df=df, ticker="NVDA", ohlcv=prices,
                            initial_capital=10000.0, threshold=0.65, rf_annual=0.05)
fig.savefig("nvda_analysis.png", dpi=150, bbox_inches="tight")
_write_metrics_csv(values, "nvda_metrics.csv", rf_annual=0.05)
```

---

## Disclaimer

This project is intended solely for academic research. Nothing in this repository constitutes financial, investment, or trading advice. Past simulated performance does not guarantee future results.

---

## Configuration Reference

Key fields in `DEFAULT_CONFIG` (all optional — defaults shown):

| Field | Default | Description |
|-------|---------|-------------|
| `deep_think_provider` | `"anthropic"` | Manager layer LLM provider |
| `deep_think_llm` | `"claude-opus-4-6"` | Manager layer model name |
| `quick_think_provider` | `"google"` | Analyst / researcher / trader layer provider |
| `quick_think_llm` | `"gemini-3-flash-preview"` | Analyst layer model name |
| `backend_url` | `None` | Base URL for OpenAI-compatible endpoints (Ollama, vLLM) |
| `parallel_analysts` | `None` | `None` = auto-detect, `True` = force parallel, `False` = force sequential |
| `max_debate_rounds` | `1` | Bull / Bear debate rounds |
| `max_risk_discuss_rounds` | `1` | Risk team debate rounds |
| `output_language` | `"English"` | Language for analyst reports and final decision |
| `run_mode` | `"single"` | `"single"` for one-day analysis, `"backtest"` for multi-day |

**`parallel_analysts` auto-detection:**
- Cloud providers (`google`, `anthropic`, `openai`, `xai`, `openrouter`) → **parallel by default**: all 4 analysts (market / social / news / fundamentals) run concurrently in separate threads, each with its own isolated message history.
- Local providers (`ollama`, `huggingface`) → **sequential by default**: concurrent local inference is hardware-limited and can cause contention.
- Override anytime: `config["parallel_analysts"] = True` or `False`.

> **Note:** The risk-analyst debate (Aggressive → Conservative → Neutral) always runs sequentially regardless of this setting. Each analyst must be able to rebut the previous speaker's argument within the same round, which requires the original sequential ordering.

---

## Key Modifications

### FinBERT Sentiment Preprocessing

Before passing social media content (Reddit, StockTwits) to the LLM, all posts are labeled with `ProsusAI/finbert`, a financial domain BERT model fine-tuned for sentiment classification. The raw posts are aggregated into a structured Bullish/Bearish/Neutral distribution summary, compressing noisy unstructured input into a concise signal. This reduces token cost and prevents the LLM from anchoring on emotionally charged language.

### Causal Memory System

The memory system is built on [MemPalace](https://github.com/milla-jovovich/mempalace), using ChromaDB for vector storage and SQLite for structured numerical records. Unstructured memories (analyst reflections, sentiment summaries) are embedded with `BAAI/bge-base-en-v1.5` for semantic retrieval.

A strict causal isolation mechanism prevents future information leakage: every read operation filters on `valid_from <= analysis_date`, and reflection memories written on day T are assigned `valid_from = T+1`, so they only become retrievable on subsequent days. This ensures that in backtest mode, agents reason solely on information available as of the analysis date.

### Backtesting Engine

A full backtesting pipeline supports batch analysis over arbitrary date ranges and multiple tickers. Signals are converted to positions via a bounded-stack strategy (matching the original TradingAgents paper) and a confidence-weighted variant. Performance is benchmarked against Buy & Hold, SMA, MACD, KDJ+RSI, and ZMR baselines, with Sharpe ratio, max drawdown, and cumulative return reported for each. Output includes a four-panel chart and a metrics CSV generated automatically after each backtest run.

---

## Roadmap

### Expanded Fundamental Data with FinBERT Labeling

The next step is to bring FOMC meeting minutes, 10-K, 10-Q, and 8-K filings into the agent pipeline using the same preprocessing approach: FinBERT labels each document chunk before it is stored in the memory system. This allows the Fundamentals Analyst to retrieve semantically relevant historical filings while the causal isolation mechanism continues to prevent future data from leaking into past analysis dates.

### Portfolio Optimization and Position Sizing

Two planned enhancements address position sizing and signal calibration. The portfolio optimizer (Modification G) will construct Markowitz and risk-parity weights across multiple tickers simultaneously, using LLM-derived confidence scores to set expected return direction while relying entirely on historical price covariance for risk estimation. A companion isotonic regression calibrator will remap raw confidence scores to empirical accuracy using accumulated backtest outcomes, feeding better-calibrated inputs to the optimizer.

The budget manager (Modification H) extends the current five-tier signal to a six-tier rating (All In / Buy / Overweight / Hold / Underweight / Sell) with rule-based position sizing: each rating maps to a fixed fraction of available capital with hard caps, and All In triggers only when a strict set of confluence conditions are met. This layer is active only in backtest mode; single-day analysis is unaffected.

---

## Citation

If you use this project, please also cite the original works it builds on:

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}
}
```

```
MemPalace — milla-jovovich
https://github.com/milla-jovovich/mempalace
MIT License
```
