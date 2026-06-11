"""Run recording and replay for MANTRA.

Captures every intermediate step of one analysis day — per-agent inputs and
outputs, LLM turns with tool-call names, tool result texts, memory retrievals,
and pipeline quality metrics (social signal-to-noise, news item counts, 10-K
availability) — into one JSON file:

    runs/{TICKER}/{YYYY-MM-DD}.json

The record serves three purposes:
  1. Live observability: CLI/GUI read it (it is rewritten atomically on every
     event) to render per-agent monitors while a run or backtest progresses.
  2. Replay cache: re-running the same (ticker, date, config digest) can skip
     the LLM pipeline entirely and return the recorded result.
  3. Post-hoc debugging: did each agent receive its upstream reports, did it
     retrieve the right memories, which tools did it actually call?

Privacy/size note: prompts are NOT stored verbatim. Agent inputs are stored as
structured summaries (state-field heads, memory hits, tool names); tool result
texts are stored in full because they are the data under inspection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HEAD = 300  # chars kept for text heads in summaries


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _head(text: Any, n: int = _HEAD) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[:n] + " …"


def config_digest(config: dict) -> str:
    """Stable digest of the config fields that affect analysis conclusions."""
    relevant = {
        k: config.get(k)
        for k in (
            "deep_think_provider", "deep_think_llm",
            "quick_think_provider", "quick_think_llm",
            "judge_iterations", "max_risk_discuss_rounds",
            "data_vendors", "tool_vendors", "output_language",
            "memory_min_similarity", "social_nli_filter",
        )
    }
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class RunRecorder:
    """Thread-safe recorder for one (ticker, date) analysis run."""

    def __init__(
        self,
        root: str,
        ticker: str,
        date: str,
        digest: str,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._path = Path(root) / ticker.upper() / f"{date}.json"
        self._data: dict = {
            "ticker": ticker.upper(),
            "date": date,
            "config_digest": digest,
            "started_at": _now(),
            "finished_at": None,
            "status": "running",
            "agents": {},        # name -> {status, llm_turns, tool_calls, inputs, output}
            "memory_events": [],
            "metrics": {},
            "final": None,
        }
        if enabled:
            self._save()

    # -- persistence ----------------------------------------------------

    def _save(self) -> None:
        """Atomic write so concurrent readers never see a torn file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1, default=str)
            os.replace(tmp, self._path)
        except Exception:
            logger.warning("RunRecorder save failed (%s)", self._path, exc_info=True)

    @property
    def path(self) -> Path:
        return self._path

    # -- agent lifecycle -------------------------------------------------

    def _agent(self, name: str) -> dict:
        return self._data["agents"].setdefault(name, {
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "inputs": {},
            "llm_turns": [],
            "tool_calls": [],
            "memory": [],
            "output": {},
        })

    def agent_start(self, name: str, inputs: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            a = self._agent(name)
            a["status"] = "in_progress"
            a["started_at"] = _now()
            if inputs:
                a["inputs"] = {k: _head(v) for k, v in inputs.items()}
            self._save()

    def agent_finish(self, name: str, output: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            a = self._agent(name)
            a["status"] = "completed"
            a["finished_at"] = _now()
            if output:
                a["output"] = {
                    k: {"chars": len(str(v or "")), "text": str(v or "")}
                    for k, v in output.items()
                }
            self._save()

    def llm_turn(self, agent: str, content: Any, tool_call_names: list[str]) -> None:
        """One LLM response inside a ReAct loop: content head + tools it invoked."""
        if not self.enabled:
            return
        with self._lock:
            self._agent(agent)["llm_turns"].append({
                "at": _now(),
                "content_head": _head(content),
                "tool_calls": tool_call_names,
            })
            self._save()

    def tool_call(self, agent: str, tool: str, args: dict, result: Any) -> None:
        if not self.enabled:
            return
        text = str(result or "")
        with self._lock:
            self._agent(agent)["tool_calls"].append({
                "at": _now(),
                "tool": tool,
                "args": {k: _head(v, 120) for k, v in (args or {}).items()},
                "result_chars": len(text),
                "result_text": text,
            })
            self._save()

    def memory_event(
        self, agent: str, room: str, query: str, hits: list[dict]
    ) -> None:
        if not self.enabled:
            return
        event = {
            "at": _now(),
            "agent": agent,
            "room": room,
            "query_head": _head(query),
            "hits": [
                {"similarity": h.get("similarity"),
                 "trade_date": h.get("trade_date"),
                 "text_head": _head(h.get("text"))}
                for h in hits
            ],
        }
        with self._lock:
            self._data["memory_events"].append(event)
            agent_rec = self._data["agents"].get(agent)
            if agent_rec is not None:
                agent_rec["memory"].append(event)
            self._save()

    # -- metrics ----------------------------------------------------------

    def metric(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data["metrics"][name] = value
            self._save()

    def bump_metric(self, name: str, key: str, n: int) -> None:
        """Atomically accumulate metrics[name][key] += n (e.g. news counts)."""
        if not self.enabled:
            return
        with self._lock:
            bucket = self._data["metrics"].setdefault(name, {})
            bucket[key] = bucket.get(key, 0) + n
            self._save()

    def get_metrics(self) -> dict:
        return dict(self._data["metrics"])

    # -- finalization ------------------------------------------------------

    def finish(self, signal: dict, reports: dict, decision_text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data["final"] = {
                "signal": signal,
                "reports": reports,
                "decision_text": decision_text,
            }
            self._data["status"] = "completed"
            self._data["finished_at"] = _now()
            self._save()

    def fail(self, error: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data["status"] = "error"
            self._data["error"] = _head(error, 2000)
            self._data["finished_at"] = _now()
            self._save()


# ---------------------------------------------------------------------------
# Active-recorder registry (one analysis runs at a time per process; analyst
# worker threads inherit the module-level current recorder).
# ---------------------------------------------------------------------------

_current: Optional[RunRecorder] = None
_NULL = RunRecorder(root=".", ticker="_", date="_", digest="_", enabled=False)


def set_recorder(rec: Optional[RunRecorder]) -> None:
    global _current
    _current = rec


def get_recorder() -> RunRecorder:
    """Return the active recorder, or a no-op recorder when none is active."""
    return _current if _current is not None else _NULL


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def load_record(root: str, ticker: str, date: str) -> Optional[dict]:
    """Load a completed run record, or None if absent/unreadable."""
    path = Path(root) / ticker.upper() / f"{date}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def find_replayable(root: str, ticker: str, date: str, digest: str) -> Optional[dict]:
    """Return a completed record matching the config digest, else None."""
    rec = load_record(root, ticker, date)
    if (
        rec
        and rec.get("status") == "completed"
        and rec.get("config_digest") == digest
        and rec.get("final")
    ):
        return rec
    return None


# ---------------------------------------------------------------------------
# Small helpers used by hook sites
# ---------------------------------------------------------------------------

def count_news_items(result: Any) -> int:
    """Best-effort count of news articles in a tool result.

    Handles the two shapes our news tools return: markdown with one "### "
    heading per article (yfinance path) and Alpha Vantage NEWS_SENTIMENT JSON
    with a "feed" list.
    """
    text = str(result or "")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("feed"), list):
            return len(parsed["feed"])
    except (json.JSONDecodeError, TypeError):
        pass
    return text.count("### ")
