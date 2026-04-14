import questionary
from datetime import datetime as _dt
from typing import List, Optional, Tuple, Dict

from rich.console import Console
from rich.panel import Panel

from cli.models import AnalystType
from tradingagents.llm_clients.model_catalog import get_model_options

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts(
    run_mode: str = "single",
    analysis_date: str = None,
) -> List[AnalystType]:
    """Select analysts using an interactive checkbox.

    All analysts including Social are always selectable.  When running in
    backtest mode or past-date single-day mode a note is shown before the
    checkbox informing the user that live social posts will be used instead
    of historically accurate data for that date.
    """
    today = _dt.now().strftime("%Y-%m-%d")
    social_live = run_mode == "single" and analysis_date == today

    if not social_live:
        if run_mode != "single":
            context = "backtest mode"
        else:
            context = f"past-date single-day mode ({analysis_date})"
        console.print(
            Panel(
                "[bold yellow]Note — Social Media Analyst[/bold yellow]\n\n"
                "Reddit and StockTwits APIs only expose the most recent posts "
                "and cannot be queried for a specific historical date.  If you "
                "include the Social Media Analyst, it will receive today's live "
                "sentiment instead of sentiment from the simulated date.\n\n"
                "This is a current data-source limitation, not a design flaw. "
                "Pairing MANTRA with a dedicated historical social-data pipeline "
                "would fully unlock this analyst's value in backtests.\n\n"
                f"[dim]Current mode: {context}[/dim]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """Fetch available models from the OpenRouter API."""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_openrouter_model() -> str:
    """Select an OpenRouter model from the newest available, or enter a custom ID."""
    models = _fetch_openrouter_models()

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "Select OpenRouter Model (latest available):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask().strip()

    return choice


def fetch_ollama_models(base_url: str) -> list[str]:
    """Fetch available model tags from a local Ollama instance.

    Returns a list of model name strings, or an empty list if unreachable.
    """
    import json
    import urllib.error
    import urllib.request

    # base_url is like "http://localhost:11434/v1"; tags live at /api/tags
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    tags_url = f"{host}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def _select_ollama_model(base_url: str | None, label: str) -> str:
    """Show a dynamic list of local Ollama models for selection."""
    models = fetch_ollama_models(base_url or "http://localhost:11434/v1")

    if not models:
        console.print(
            "[yellow]Warning: Could not reach Ollama at "
            f"{base_url or 'http://localhost:11434/v1'}. "
            "Falling back to manual entry.[/yellow]"
        )
        return questionary.text(
            f"Enter exact Ollama model tag for [{label}]:",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model tag.",
        ).ask().strip()

    choices = [questionary.Choice(name, value=name) for name in models]
    choices.append(questionary.Choice("(enter manually)", value="__manual__"))

    choice = questionary.select(
        f"Select Your [{label}] (fetched from local Ollama):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print(f"\n[red]No model selected for [{label}]. Exiting...[/red]")
        exit(1)

    if choice == "__manual__":
        return questionary.text(
            "Enter exact Ollama model tag:",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model tag.",
        ).ask().strip()

    return choice


def select_shallow_thinking_agent(provider, url: str | None = None) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    if provider.lower() == "openrouter":
        return select_openrouter_model()

    if provider.lower() == "ollama":
        return _select_ollama_model(url, "Quick-Thinking LLM Engine")

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, "quick")
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]No shallow thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider, url: str | None = None) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    if provider.lower() == "openrouter":
        return select_openrouter_model()

    if provider.lower() == "ollama":
        return _select_ollama_model(url, "Deep-Thinking LLM Engine")

    choice = questionary.select(
        "Select Your [Deep-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, "deep")
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No deep thinking llm engine selected. Exiting...[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Google", None),  # google-genai SDK manages its own endpoint
        ("Anthropic", "https://api.anthropic.com/"),
        ("xAI", "https://api.x.ai/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("HuggingFace", None),  # HF SDK uses its own default endpoint; override via backend_url for vLLM
        ("Ollama", "http://localhost:11434/v1"),
    ]
    
    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]no OpenAI backend selected. Exiting...[/red]")
        exit(1)
    
    display_name, url = choice
    print(f"You selected: {display_name}\tURL: {url}")

    return display_name, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """Ask for Anthropic effort level.

    Controls token usage and response thoroughness on Claude 4.5+ and 4.6 models.
    """
    return questionary.select(
        "Select Effort Level:",
        choices=[
            questionary.Choice("High (recommended)", "high"),
            questionary.Choice("Medium (balanced)", "medium"),
            questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """Ask for Gemini thinking configuration.

    Returns thinking_level: "high" or "minimal".
    Client maps to appropriate API param based on model series.
    """
    return questionary.select(
        "Select Thinking Mode:",
        choices=[
            questionary.Choice("Enable Thinking (recommended)", "high"),
            questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()


def select_analyst_llm_config() -> dict:
    """Select analyst (quick-thinking) LLM: provider then model.

    Analysts, Researchers, and Trader all use this client.

    Returns a dict with keys: provider, model, url,
        google_thinking_level, openai_reasoning_effort
    """
    provider_display, url = select_llm_provider()
    model = select_shallow_thinking_agent(provider_display, url)

    google_thinking_level = None
    openai_reasoning_effort = None

    if provider_display.lower() == "google":
        google_thinking_level = ask_gemini_thinking_config()
    elif provider_display.lower() == "openai":
        openai_reasoning_effort = ask_openai_reasoning_effort()

    return {
        "provider": provider_display.lower(),
        "model": model,
        "url": url,
        "google_thinking_level": google_thinking_level,
        "openai_reasoning_effort": openai_reasoning_effort,
    }


def select_manager_llm_config() -> dict:
    """Select manager (deep-thinking) LLM: provider then model.

    Research Manager and Portfolio Manager use this client.

    Returns a dict with keys: provider, model, url,
        anthropic_effort, google_thinking_level, openai_reasoning_effort
    """
    provider_display, url = select_llm_provider()
    model = select_deep_thinking_agent(provider_display, url)

    anthropic_effort = None
    google_thinking_level = None
    openai_reasoning_effort = None

    if provider_display.lower() == "anthropic":
        anthropic_effort = ask_anthropic_effort()
    elif provider_display.lower() == "google":
        google_thinking_level = ask_gemini_thinking_config()
    elif provider_display.lower() == "openai":
        openai_reasoning_effort = ask_openai_reasoning_effort()

    return {
        "provider": provider_display.lower(),
        "model": model,
        "url": url,
        "anthropic_effort": anthropic_effort,
        "google_thinking_level": google_thinking_level,
        "openai_reasoning_effort": openai_reasoning_effort,
    }


def ask_output_language() -> str:
    """Ask for report output language."""
    choice = questionary.select(
        "Select Output Language:",
        choices=[
            questionary.Choice("English (default)", "English"),
            questionary.Choice("Chinese (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean (한국어)", "Korean"),
            questionary.Choice("Hindi (हिन्दी)", "Hindi"),
            questionary.Choice("Spanish (Español)", "Spanish"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("French (Français)", "French"),
            questionary.Choice("German (Deutsch)", "German"),
            questionary.Choice("Arabic (العربية)", "Arabic"),
            questionary.Choice("Russian (Русский)", "Russian"),
            questionary.Choice("Custom language", "custom"),
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask().strip()

    return choice


# ---------------------------------------------------------------------------
# Run mode, multi-ticker, and date-range helpers (for backtest CLI flow)
# ---------------------------------------------------------------------------

def select_run_mode() -> str:
    """Select analysis run mode.

    Returns one of:
        "single"           — single stock, single day
        "backtest_single"  — single stock, multi-day backtest
        "backtest_multi"   — multiple stocks, multi-day backtest
    """
    RUN_MODES = [
        ("Single stock, single day   — one-time analysis", "single"),
        ("Single stock, multi-day    — backtest over a date range", "backtest_single"),
        ("Multi-stock, multi-day     — portfolio backtest over a date range", "backtest_multi"),
    ]
    choice = questionary.select(
        "Select Run Mode:",
        choices=[
            questionary.Choice(display, value=value) for display, value in RUN_MODES
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()
    if choice is None:
        console.print("\n[red]No run mode selected. Exiting...[/red]")
        exit(1)
    return choice


def get_tickers_multi() -> list[str]:
    """Prompt for comma-separated ticker symbols (multi-stock mode).

    Returns a list of normalised ticker strings.
    """
    raw = questionary.text(
        "Enter ticker symbols separated by commas (e.g. AAPL, NVDA, MSFT):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter at least one ticker.",
        style=questionary.Style([
            ("text", "fg:green"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()
    if not raw:
        console.print("\n[red]No tickers provided. Exiting...[/red]")
        exit(1)
    tickers = [normalize_ticker_symbol(t) for t in raw.split(",") if t.strip()]
    if not tickers:
        console.print("\n[red]No valid tickers parsed. Exiting...[/red]")
        exit(1)
    return tickers


def get_start_date() -> str:
    """Prompt for backtest start date. Default: 3 months ago."""
    import re
    from datetime import datetime, timedelta

    default = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    def _valid(s: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s.strip()):
            return False
        try:
            datetime.strptime(s.strip(), "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        f"Enter backtest start date (YYYY-MM-DD):",
        default=default,
        validate=lambda x: _valid(x) or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style([
            ("text", "fg:green"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()
    if not date:
        console.print("\n[red]No start date provided. Exiting...[/red]")
        exit(1)
    return date.strip()


def get_end_date() -> str:
    """Prompt for backtest end date. Default: today."""
    import re
    from datetime import datetime

    default = datetime.now().strftime("%Y-%m-%d")

    def _valid(s: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s.strip()):
            return False
        try:
            datetime.strptime(s.strip(), "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        f"Enter backtest end date (YYYY-MM-DD):",
        default=default,
        validate=lambda x: _valid(x) or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style([
            ("text", "fg:green"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()
    if not date:
        console.print("\n[red]No end date provided. Exiting...[/red]")
        exit(1)
    return date.strip()
