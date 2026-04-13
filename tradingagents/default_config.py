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
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
