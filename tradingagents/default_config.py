import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "memory_palace_path": os.getenv("TRADINGAGENTS_MEMORY_PATH", "./memory"),
    # Run mode controls the memory system:
    #   "single"   — one-day analysis, TradingMemoryStore disabled (no disk writes)
    #   "backtest" — multi-day loop via run_backtest(), persistent ChromaDB + SQLite enabled
    "run_mode": "single",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    # llm_provider remains as a backward compatibility fallback.
    "llm_provider": "google",           # fallback; prefer deep/quick_think_provider below

    # --- Manager layer (Research Manager / Portfolio Manager) ---
    # Always uses Claude Opus 4.6. No local model option: these are the final decision
    # nodes and require the strongest available reasoning.
    "deep_think_provider": "anthropic",
    "deep_think_llm": "claude-opus-4-6",

    # --- Analyst / Researcher / Trader layer ---
    # Default: Gemini 3 Flash (fast, cost-effective; same API key as analysts).
    # Local alternative: set quick_think_provider="ollama", quick_think_llm="Gemma4-Quant-31B"
    "quick_think_provider": "google",
    "quick_think_llm": "gemini-3-flash-preview",

    "backend_url": None,                # Leave None for cloud providers; set for Ollama/self-hosted
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    # judge_iterations controls how many Judge critique cycles occur before
    # the debate is handed to the Research Manager.
    #   1 → 5 API calls  (Bull+Bear initial → Judge → Bull+Bear response)
    #   2 → 8 API calls  (adds one more Judge + researcher round)
    #   3 → 11 API calls (adds another Judge + researcher round)
    "judge_iterations": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Parallel analyst execution.
    # None  → auto: True for cloud providers, False for local (ollama / huggingface)
    #         because running multiple local models concurrently is hardware-limited.
    # True  → always parallel (use with caution on local models).
    # False → always sequential.
    "parallel_analysts": None,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        # News default is alpha_vantage: yfinance news has NO date parameters
        # (always returns the latest ~20 articles), which silently breaks any
        # historical-date analysis. Requires ALPHA_VANTAGE_API_KEY.
        "news_data": "alpha_vantage",        # Options: alpha_vantage, yfinance
    },
    # Minimum cosine similarity for memory retrieval hits (see TradingMemoryStore).
    "memory_min_similarity": 0.45,
    # Social subject filtering: zero-shot NLI check that a post is primarily
    # about the target ticker (runs after the free rule-based prefilter).
    "social_nli_filter": True,
    "social_nli_model": "typeform/distilbert-base-uncased-mnli",
    # Raw social post archival — OFF locally (no bulk storage); flip to True
    # on a cloud deployment to start accumulating a historical corpus.
    "archive_raw_social": False,
    "raw_social_archive_dir": "./data/raw_social",
    # Run recording & replay. Every run writes its intermediate steps to
    # runs/{ticker}/{date}.json (agent monitor + debugging). When a completed
    # record with the same config digest exists, reuse_cached_run returns it
    # without re-invoking the LLM pipeline (a "Fresh Run" sets this False).
    "record_runs": True,
    "reuse_cached_run": True,
    "runs_dir": os.getenv("TRADINGAGENTS_RUNS_DIR", "./runs"),
    # 10-K RAG (SEC EDGAR; free, no key). The index is durable reference data
    # kept SEPARATE from the experiential memory store — see docs.
    "filings_index_dir": "./data/filings_index",
    # A filing stays retrievable for this long after filed_date (3 years).
    "filing_validity_days": 1095,
    # Earnings call transcripts stay retrievable for one year (guidance is
    # superseded quarterly; YoY narrative comparison keeps value).
    "transcript_validity_days": 365,
    # API Ninjas key for earnings call transcripts (free tier; env fallback).
    "api_ninjas_key": os.getenv("API_NINJAS_KEY"),
    # SEC requires a User-Agent identifying the requester.
    "edgar_user_agent": os.getenv("EDGAR_USER_AGENT"),
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
