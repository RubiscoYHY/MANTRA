"""
tradingagents/agents/utils/memory_store.py

TradingMemoryStore — unified memory interface for TradingAgents.

All writes from Analyst nodes, Reflector, and Portfolio Manager are routed
through this class. Callers only call typed methods and never touch ChromaDB
or the SQLite knowledge graph directly.

Writer → Room → Backend routing:
    Social Analyst      → room "sentiment"     → ChromaDB   expires trade_date + 3d
    News Analyst        → room "news"          → ChromaDB   expires trade_date + 3d
    Market Analyst      → room "market"        → ChromaDB   expires trade_date + 7d
    Fundamentals Analyst→ room "fundamentals"  → ChromaDB   expires trade_date + 365d
    Portfolio Manager   → room "lessons"       → ChromaDB   never expires,
                                                             valid_from = trade_date + 1d
    Reflector           → KG triple            → SQLite KG  single-day fact
    Data pipeline       → KG triple            → SQLite KG  permanent historical fact

Causal isolation
----------------
All reads filter by valid_from <= _analysis_date, so no future memory can
ever surface. The analysis date is set once per day by the trading graph via
set_analysis_date(); individual agents never touch it.

    # In trading_graph.py, before running a day's analysis:
    memory_store.set_analysis_date(trade_date)

    # In any agent node — no date argument needed:
    results = memory_store.retrieve_similar_sentiment(ticker, query)

Lessons from day T receive valid_from = T+1, so they cannot influence an
analysis run on the same day T (they become visible only from T+1 onwards).

Disabled mode
-------------
When enabled=False all methods are no-ops. Safe to construct the object
unconditionally; actual storage is only created on the first real call when
enabled=True.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Return the best available torch device: cuda, mps, or cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class _BGEEmbeddingFunction:
    """
    ChromaDB embedding backend using BAAI/bge-base-en-v1.5.

    BGE uses asymmetric retrieval: documents are encoded without a prefix,
    queries prepend a fixed instruction string. ChromaDB's __call__ is
    invoked for document upsert only; _search_text() calls encode_query()
    directly and passes query_embeddings to col.query(), bypassing __call__
    on the query path so the prefix is applied correctly.

    Lazy-loaded: the SentenceTransformer model is not imported or downloaded
    until the first call to __init__, which is itself deferred until the
    first ChromaDB collection access (i.e. only in backtest mode).
    """

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device or _detect_device())

    def __call__(self, input):  # noqa: A002
        """Encode documents without prefix. Called by ChromaDB during upsert."""
        vecs = self._model.encode(list(input), normalize_embeddings=True)
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode one query string with BGE's asymmetric instruction prefix."""
        vec = self._model.encode(
            self._QUERY_PREFIX + query, normalize_embeddings=True
        )
        return vec.tolist()


# Sentinel for "never expires". Stored as integer YYYYMMDD (99991231) so
# ChromaDB $lte/$gte operators (numeric-only) work correctly.
_NEVER_EXPIRES = 99991231


def _date_to_int(date_str: str) -> int:
    """Convert ISO date string to integer YYYYMMDD for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))

# Validity periods per room, in days. None = never expires.
_EXPIRY_DAYS: dict[str, Optional[int]] = {
    "sentiment":    3,
    "news":         3,
    "market":       7,
    "fundamentals": 365,
    "lessons":      None,
    # Role-specific reflection rooms (replaces FinancialSituationMemory / BM25)
    "reflections_bull":               None,
    "reflections_bear":               None,
    "reflections_trader":             None,
    "reflections_invest_judge":       None,
    "reflections_portfolio_manager":  None,
}


def _expiry_date(room: str, from_date: str) -> int:
    """Return the expiry date as integer YYYYMMDD for ChromaDB numeric filtering."""
    days = _EXPIRY_DAYS.get(room)
    if days is None:
        return _NEVER_EXPIRES
    base = datetime.fromisoformat(from_date)
    return _date_to_int((base + timedelta(days=days)).strftime("%Y-%m-%d"))


def _drawer_id(wing: str, room: str, trade_date: str) -> str:
    """Deterministic Drawer ID. Upserting with the same ID overwrites cleanly."""
    key = f"{wing}_{room}_{trade_date}"
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return f"drawer_{wing}_{room}_{h}"


class TradingMemoryStore:
    """
    Unified read/write memory interface for TradingAgents.

    Typical setup in trading_graph.py:
        store = TradingMemoryStore(
            palace_path=config["memory_palace_path"],
            enabled=config.get("use_memory", False),
        )
        store.set_analysis_date(trade_date)   # once per day
        # inject store into agent node constructors

    Agents call typed methods — no date argument required:
        store.store_sentiment_summary(ticker, summary_text)
        results = store.retrieve_similar_sentiment(ticker, query)
    """

    def __init__(self, palace_path: str, enabled: bool = False):
        self._palace_path = palace_path
        self._enabled = enabled
        self._analysis_date: Optional[str] = None  # set by trading_graph per day
        self._col = None   # lazy: chromadb collection
        self._ef: Optional[_BGEEmbeddingFunction] = None  # lazy: BGE embedding function
        self._kg = None    # lazy: KnowledgeGraph

    # ------------------------------------------------------------------
    # Analysis date control (called by trading_graph.py, not by agents)
    # ------------------------------------------------------------------

    def set_analysis_date(self, date: str) -> None:
        """
        Set the current analysis date. Must be called by trading_graph.py
        at the start of each day's analysis before any agent runs.

        All read methods use this date as the causal isolation gate
        (valid_from <= analysis_date). All write methods use it as trade_date.

        Args:
            date: ISO date string, e.g. "2024-01-15".
        """
        self._analysis_date = date
        logger.debug("MemoryStore analysis date set to %s", date)

    def _require_date(self) -> str:
        """Return the current analysis date, raising if not set."""
        if self._analysis_date is None:
            raise RuntimeError(
                "TradingMemoryStore.set_analysis_date() must be called before "
                "any read or write operation."
            )
        return self._analysis_date

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_col(self):
        """Return (or create) the ChromaDB collection backed by BGE embeddings."""
        if self._col is None:
            import chromadb
            self._ef = _BGEEmbeddingFunction()
            client = chromadb.PersistentClient(path=self._palace_path)
            self._col = client.get_or_create_collection(
                "mempalace_drawers",
                embedding_function=self._ef,
            )
        return self._col

    def _get_kg(self):
        """Return (or create) the SQLite KnowledgeGraph."""
        if self._kg is None:
            from tradingagents.agents.utils.knowledge_graph import KnowledgeGraph
            kg_path = str(Path(self._palace_path) / "kg.sqlite3")
            self._kg = KnowledgeGraph(db_path=kg_path)
        return self._kg

    # ------------------------------------------------------------------
    # Internal: ChromaDB write
    # ------------------------------------------------------------------

    def _store_text(
        self,
        ticker: str,
        room: str,
        trade_date: str,
        content: str,
        valid_from: str,
        expires_at: str,
        writer: str,
    ) -> bool:
        """
        Write one Drawer to ChromaDB with extended metadata schema.

        The Drawer ID is deterministic (wing + room + trade_date), so calling
        this twice for the same ticker/room/date replaces the previous entry.
        """
        if not self._enabled:
            return False
        try:
            col = self._get_col()
            wing = ticker.lower()
            doc_id = _drawer_id(wing, room, trade_date)
            now = datetime.now().isoformat()
            metadata = {
                # MemPalace-compatible fields
                "wing":          wing,
                "room":          room,
                "source_file":   f"{wing}_{room}_{trade_date}",
                "chunk_index":   0,
                "added_by":      writer,
                "filed_at":      now,
                # Extended fields for causal isolation and expiry
                "trade_date":    trade_date,
                "valid_from":    _date_to_int(valid_from),
                "expires_at":    expires_at,  # already int from _expiry_date()
                "recorded_at":   now,
            }
            col.upsert(documents=[content], ids=[doc_id], metadatas=[metadata])
            logger.debug("Stored %s/%s/%s by %s", wing, room, trade_date, writer)
            return True
        except Exception as e:
            logger.warning(
                "MemoryStore write failed (%s/%s/%s): %s", ticker, room, trade_date, e
            )
            return False

    # ------------------------------------------------------------------
    # Internal: ChromaDB read with causal isolation + expiry filter
    # ------------------------------------------------------------------

    def _search_text(
        self,
        query: str,
        wing: Optional[str],
        room: str,
        as_of: str,
        n_results: int,
    ) -> list[dict]:
        """
        L3 semantic search with two mandatory filters:
            valid_from <= as_of   (causal isolation)
            expires_at >= as_of   (exclude stale memories)

        Returns a list of result dicts, sorted by similarity descending.
        Returns [] when disabled or on any error.
        """
        if not self._enabled:
            return []
        try:
            col = self._get_col()

            # Build filter. $and requires at least two conditions.
            # valid_from and expires_at are stored as int YYYYMMDD so that
            # ChromaDB $lte/$gte operators (numeric-only) work correctly.
            as_of_int = _date_to_int(as_of)
            conditions: list[dict] = [
                {"room":       {"$eq": room}},
                {"valid_from": {"$lte": as_of_int}},
                {"expires_at": {"$gte": as_of_int}},
            ]
            if wing:
                conditions.append({"wing": {"$eq": wing}})

            where = {"$and": conditions}

            # Use encode_query() so the BGE asymmetric prefix is applied
            # to the query vector while stored documents remain prefix-free.
            query_vec = self._ef.encode_query(query)
            results = col.query(
                query_embeddings=[query_vec],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "text":       doc,
                    "wing":       meta.get("wing"),
                    "room":       meta.get("room"),
                    "trade_date": meta.get("trade_date"),
                    "valid_from": meta.get("valid_from"),
                    "similarity": round(1.0 - dist, 3),
                    "writer":     meta.get("added_by"),
                })
            return hits

        except Exception as e:
            logger.warning(
                "MemoryStore search failed (%s/%s as_of=%s): %s", wing, room, as_of, e
            )
            return []

    # ==================================================================
    # PUBLIC WRITE API
    # All write methods use _require_date() as trade_date automatically.
    # No date argument needed from calling agents.
    # ==================================================================

    def store_sentiment_summary(self, ticker: str, summary: str) -> bool:
        """
        Social Analyst: FinBERT-aggregated sentiment summary.
        Backend: ChromaDB. Room: "sentiment". Expires in 3 days.
        """
        if not self._enabled:
            return False
        td = self._require_date()
        return self._store_text(
            ticker=ticker, room="sentiment", trade_date=td,
            content=summary,
            valid_from=td,
            expires_at=_expiry_date("sentiment", td),
            writer="social_analyst",
        )

    def store_news_summary(self, ticker: str, summary: str) -> bool:
        """
        News Analyst: news digest for the day.
        Backend: ChromaDB. Room: "news". Expires in 3 days.
        """
        if not self._enabled:
            return False
        td = self._require_date()
        return self._store_text(
            ticker=ticker, room="news", trade_date=td,
            content=summary,
            valid_from=td,
            expires_at=_expiry_date("news", td),
            writer="news_analyst",
        )

    def store_market_summary(self, ticker: str, summary: str) -> bool:
        """
        Market Analyst: technical analysis summary.
        Backend: ChromaDB. Room: "market". Expires in 7 days.
        """
        if not self._enabled:
            return False
        td = self._require_date()
        return self._store_text(
            ticker=ticker, room="market", trade_date=td,
            content=summary,
            valid_from=td,
            expires_at=_expiry_date("market", td),
            writer="market_analyst",
        )

    def store_fundamentals(self, ticker: str, report: str) -> bool:
        """
        Fundamentals Analyst: financial fundamentals report.
        Backend: ChromaDB. Room: "fundamentals". Expires in 365 days.
        """
        if not self._enabled:
            return False
        td = self._require_date()
        return self._store_text(
            ticker=ticker, room="fundamentals", trade_date=td,
            content=report,
            valid_from=td,
            expires_at=_expiry_date("fundamentals", td),
            writer="fundamentals_analyst",
        )

    def store_lesson(
        self,
        ticker: str,
        lesson: str,
        decision: str = "",
        outcome: str = "",
    ) -> bool:
        """
        Portfolio Manager: post-trade lesson derived from this day's decision.

        Backend: ChromaDB. Room: "lessons". Never expires.
        Causal isolation: valid_from = analysis_date + 1 day, so this lesson
        cannot influence an analysis run on the same day it is recorded.

        Args:
            lesson:   What should be done differently next time.
            decision: The decision made today (BUY / SELL / HOLD + rationale).
            outcome:  Qualitative outcome (filled in together with actual_return).
        """
        if not self._enabled:
            return False
        td = self._require_date()
        next_day = (
            datetime.fromisoformat(td) + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        content = lesson
        if decision or outcome:
            parts = []
            if decision:
                parts.append(f"Decision: {decision}")
            if outcome:
                parts.append(f"Outcome: {outcome}")
            parts.append(f"Lesson: {lesson}")
            content = "\n".join(parts)

        return self._store_text(
            ticker=ticker, room="lessons", trade_date=td,
            content=content,
            valid_from=next_day,        # +1 day: causal isolation gate
            expires_at=_NEVER_EXPIRES,
            writer="portfolio_manager",
        )

    def annotate_return(self, ticker: str, actual_return: float) -> bool:
        """
        Reflector: record the actual return for ticker on the current analysis date.
        Backend: SQLite KG. Fact is valid only on that single day.

        Idempotent: if a return is already recorded for this ticker+date,
        the call is a no-op (safe to call from retry logic).
        """
        if not self._enabled:
            return False
        td = self._require_date()
        try:
            kg = self._get_kg()
            subject = f"{ticker.lower()}_{td}"

            # Guard: prevent duplicate triples (KG dedup only covers valid_to IS NULL).
            existing = kg.query_entity(subject, as_of=td)
            if any(r["predicate"] == "actual_return" for r in existing):
                logger.debug("Return already annotated for %s on %s", ticker, td)
                return True

            kg.add_triple(
                subject=subject,
                predicate="actual_return",
                obj=str(actual_return),
                valid_from=td,
                valid_to=td,            # single-day fact
                confidence=1.0,
            )
            return True
        except Exception as e:
            logger.warning("annotate_return failed (%s/%s): %s", ticker, td, e)
            return False

    def store_price(self, ticker: str, close_price: float) -> bool:
        """
        Data pipeline: record the closing price for ticker on the current analysis date.
        Backend: SQLite KG. Permanent historical fact (valid_to=None).

        Idempotent: KnowledgeGraph.add_triple() deduplicates triples with
        the same (subject, predicate, object) when valid_to IS NULL.
        """
        if not self._enabled:
            return False
        td = self._require_date()
        try:
            kg = self._get_kg()
            kg.add_triple(
                subject=f"{ticker.lower()}_{td}",
                predicate="close_price",
                obj=str(close_price),
                valid_from=td,
                valid_to=None,          # permanent historical fact
                confidence=1.0,
            )
            return True
        except Exception as e:
            logger.warning("store_price failed (%s/%s): %s", ticker, td, e)
            return False

    # ==================================================================
    # PUBLIC READ API
    # All read methods use _require_date() as the causal isolation gate.
    # No as_of argument needed from calling agents.
    # ==================================================================

    def retrieve_similar_sentiment(
        self,
        ticker: str,
        query: str,
        n_results: int = 3,
    ) -> list[dict]:
        """
        L3 semantic search: find historical sentiment patterns for ticker
        similar to the current query, respecting causal isolation.

        Returns list of dicts: {text, wing, room, trade_date, valid_from,
        similarity, writer}. Empty list if disabled or no match found.
        """
        if not self._enabled:
            return []
        return self._search_text(
            query=query,
            wing=ticker.lower(),
            room="sentiment",
            as_of=self._require_date(),
            n_results=n_results,
        )

    def retrieve_sector_sentiment(
        self,
        related_tickers: list[str],
        query: str,
        n_results: int = 2,
    ) -> list[dict]:
        """
        Cross-wing L3: search historical sentiment across sector peers.

        Searches each ticker in related_tickers independently, then returns
        the top n_results hits by similarity score across all wings. The Tunnel
        structure (same room "sentiment" across wings) makes this natural.

        Args:
            related_tickers: peer tickers, e.g. ["amd", "intc"] when analysing NVDA.
            n_results:       total hits to return across all tickers combined.
        """
        if not self._enabled:
            return []
        as_of = self._require_date()
        all_hits: list[dict] = []
        for ticker in related_tickers:
            hits = self._search_text(
                query=query,
                wing=ticker.lower(),
                room="sentiment",
                as_of=as_of,
                n_results=n_results,
            )
            all_hits.extend(hits)

        all_hits.sort(key=lambda h: h["similarity"], reverse=True)
        return all_hits[:n_results]

    def retrieve_lessons(
        self,
        ticker: str,
        query: str,
        n_results: int = 3,
    ) -> list[dict]:
        """
        L3 semantic search: find relevant past lessons for ticker.

        Causal isolation is enforced automatically: lessons with valid_from
        = today are excluded (they were recorded today and must not feed back
        into today's analysis). Only lessons from previous days are returned.
        """
        if not self._enabled:
            return []
        return self._search_text(
            query=query,
            wing=ticker.lower(),
            room="lessons",
            as_of=self._require_date(),
            n_results=n_results,
        )

    def get_historical_return(self, ticker: str, date: str) -> Optional[float]:
        """
        Exact lookup: what was the actual return for ticker on a specific date?

        Unlike the write methods, this accepts an explicit date so callers
        (e.g. Reflector computing reward) can query any past date.
        Returns None if not yet annotated or store is disabled.
        """
        if not self._enabled:
            return None
        try:
            kg = self._get_kg()
            rows = kg.query_entity(f"{ticker.lower()}_{date}", as_of=date)
            for row in rows:
                if row.get("predicate") == "actual_return":
                    return float(row["object"])
        except Exception as e:
            logger.warning("get_historical_return failed (%s/%s): %s", ticker, date, e)
        return None

    def get_price(self, ticker: str, date: str) -> Optional[float]:
        """
        Exact lookup: what was the closing price for ticker on a specific date?

        Accepts an explicit date so callers can query any past date.
        Returns None if not recorded or store is disabled.
        """
        if not self._enabled:
            return None
        try:
            kg = self._get_kg()
            rows = kg.query_entity(f"{ticker.lower()}_{date}", as_of=date)
            for row in rows:
                if row.get("predicate") == "close_price":
                    return float(row["object"])
        except Exception as e:
            logger.warning("get_price failed (%s/%s): %s", ticker, date, e)
        return None

    def store_reflection(
        self,
        ticker: str,
        role: str,
        situation: str,
        recommendation: str,
    ) -> bool:
        """
        Store a post-trade reflection for a given agent role.

        Replaces FinancialSituationMemory / BM25 as the persistent reflection backend.
        Backend: ChromaDB. Room: "reflections_{role}". Never expires.
        Causal isolation: valid_from = analysis_date + 1, so this reflection
        cannot influence an analysis run on the same day it is recorded.

        Args:
            role:           Agent role key, one of: bull, bear, trader,
                            invest_judge, portfolio_manager.
            situation:      Concatenated analyst reports describing today's market.
            recommendation: LLM-generated reflection / lesson learned.
        """
        if not self._enabled:
            return False
        td = self._require_date()
        next_day = (
            datetime.fromisoformat(td) + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        room = f"reflections_{role}"
        content = f"Situation:\n{situation}\n\nLesson:\n{recommendation}"
        return self._store_text(
            ticker=ticker, room=room, trade_date=td,
            content=content,
            valid_from=next_day,
            expires_at=_NEVER_EXPIRES,
            writer=role,
        )

    def record_calibration_point(
        self,
        ticker: str,
        trade_date: str,
        signal: str,
        confidence: float,
        actual_return: float,
    ) -> bool:
        """
        Persist one calibration record to calibration.jsonl inside the palace.

        Each line is a JSON object with keys: ticker, date, signal, confidence,
        actual_return. Called by reflect_and_remember() in backtest mode.
        Records accumulate across multiple backtest runs and are used by
        G.7.4 (Platt / isotonic calibration) when 50+ samples exist.
        """
        if not self._enabled:
            return False
        import json as _json
        record = {
            "ticker":        ticker,
            "date":          trade_date,
            "signal":        signal,
            "confidence":    confidence,
            "actual_return": actual_return,
        }
        cal_path = Path(self._palace_path) / "calibration.jsonl"
        try:
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cal_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
            return True
        except Exception as e:
            logger.warning("record_calibration_point failed: %s", e)
            return False

    def load_calibration_records(self, ticker: Optional[str] = None) -> list[dict]:
        """
        Load all calibration records from calibration.jsonl.

        Args:
            ticker: If provided, filter records to this ticker (case-insensitive).

        Returns:
            List of record dicts: {ticker, date, signal, confidence, actual_return}.
            Empty list when the file does not exist or the store is disabled.
        """
        import json as _json
        cal_path = Path(self._palace_path) / "calibration.jsonl"
        if not cal_path.exists():
            return []
        records: list[dict] = []
        try:
            with open(cal_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _json.loads(line)
                    if ticker is None or rec.get("ticker", "").upper() == ticker.upper():
                        records.append(rec)
        except Exception as e:
            logger.warning("load_calibration_records failed: %s", e)
        return records

    def retrieve_reflections(
        self,
        ticker: str,
        role: str,
        query: str,
        n_results: int = 2,
    ) -> list[dict]:
        """
        Semantic search for historical reflections for a given agent role.

        Replaces FinancialSituationMemory.get_memories().
        Returns list of dicts: {text, trade_date, similarity, ...}.
        Empty list when disabled or no relevant history exists.

        Args:
            role:      Agent role key matching what was used in store_reflection().
            query:     Current market situation text used as the search query.
            n_results: Number of top matches to return.
        """
        if not self._enabled:
            return []
        return self._search_text(
            query=query,
            wing=ticker.lower(),
            room=f"reflections_{role}",
            as_of=self._require_date(),
            n_results=n_results,
        )
