"""FilingStore — vector index over SEC filing text (10-K RAG).

Deliberately a SEPARATE database from the experiential memory store
(memory/): filings are durable reference data that survive experiment
resets, while reflections are wiped and rebuilt per experiment. Both stores
share one process-wide BGE embedding instance (injected via
get_shared_embedding_fn(); never instantiate a second model).

Causal isolation: every chunk carries filed_date as integer YYYYMMDD and
searches filter filed_date <= analysis date, so a backtest can never read a
filing that did not exist yet.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

from tradingagents.agents.utils.memory_store import get_shared_embedding_fn
from tradingagents.dataflows.filings import (
    FilingRef,
    fetch_filing_text,
    find_latest_filing,
)

logger = logging.getLogger(__name__)

_CHUNK_CHARS = 1800
_CHUNK_OVERLAP = 200
_MAX_CHUNKS = 600          # hard cap per filing (~1MB of text)
_EMBED_BATCH = 32

# A filing stays retrievable for this long after its filed_date. Annual
# reports describe a business that drifts; a 10-K older than ~3 years is more
# likely to mislead than to inform.
_DEFAULT_VALIDITY_DAYS = 3 * 365

# Earnings calls age much faster than annual reports: guidance is superseded
# by the next call (~90 days), but year-over-year narrative comparison keeps
# value, so calls stay retrievable for one year. Hits carry their call date,
# letting the LLM weigh recency itself.
_DEFAULT_TRANSCRIPT_VALIDITY_DAYS = 365


def _date_to_int(date_str: str) -> int:
    return int(date_str.replace("-", ""))


def _chunk(text: str) -> list[str]:
    """Greedy fixed-size chunking on paragraph boundaries where possible."""
    chunks: list[str] = []
    pos = 0
    n = len(text)
    while pos < n and len(chunks) < _MAX_CHUNKS:
        end = min(pos + _CHUNK_CHARS, n)
        if end < n:
            # Prefer breaking at a paragraph/sentence boundary inside the tail.
            window = text[pos:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "))
            if cut > _CHUNK_CHARS // 2:
                end = pos + cut + 1
        chunks.append(text[pos:end].strip())
        if end >= n:
            # Reached the end — never emit a trailing overlap-only fragment.
            break
        pos = max(end - _CHUNK_OVERLAP, pos + 1)
    return [c for c in chunks if c]


class FilingStore:
    """Chroma-backed vector index of filing chunks (collection "filings")."""

    def __init__(
        self,
        index_path: str = "./data/filings_index",
        validity_days: int = _DEFAULT_VALIDITY_DAYS,
        transcript_validity_days: int = _DEFAULT_TRANSCRIPT_VALIDITY_DAYS,
    ):
        self._index_path = index_path
        self._validity_days = validity_days
        self._transcript_validity_days = transcript_validity_days
        self._col = None
        self._lock = threading.Lock()

    def _get_col(self):
        if self._col is not None:
            return self._col
        with self._lock:
            if self._col is None:
                import chromadb
                client = chromadb.PersistentClient(path=self._index_path)
                self._col = client.get_or_create_collection(
                    "filings", metadata={"hnsw:space": "cosine"}
                )
        return self._col

    # -- ingestion ---------------------------------------------------------

    def _is_ingested(self, accession: str) -> bool:
        got = self._get_col().get(where={"accession": accession}, limit=1)
        return bool(got["ids"])

    def ingest_text(self, ref: FilingRef, text: str) -> int:
        """Chunk, embed, and index one filing. Returns chunk count."""
        col = self._get_col()
        chunks = _chunk(text)
        if not chunks:
            return 0
        ef = get_shared_embedding_fn()
        filed_int = _date_to_int(ref.filed_date)
        for start in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[start:start + _EMBED_BATCH]
            vecs = ef(batch)
            col.upsert(
                ids=[f"{ref.accession}_{start + i}" for i in range(len(batch))],
                documents=batch,
                embeddings=vecs,
                metadatas=[{
                    "wing": ref.ticker.lower(),
                    "accession": ref.accession,
                    "form": ref.form,
                    "filed_date": filed_int,
                    "chunk_index": start + i,
                } for i in range(len(batch))],
            )
        logger.info("Indexed %s %s: %d chunks", ref.ticker, ref.form, len(chunks))
        return len(chunks)

    def ensure_ingested(self, ticker: str, as_of: str) -> Optional[FilingRef]:
        """Locate the latest 10-K filed on or before as_of and index it.

        10-K only by design: its Item 1A and footnotes are far richer than a
        10-Q's, and the 10-Q's structured numbers already reach the analyst
        through get_balance_sheet/get_income_statement/get_cashflow. The 10-Q
        fallback exists solely for companies with NO 10-K yet (recent IPOs).
        """
        ref = find_latest_filing(ticker, as_of, forms=("10-K",))
        if ref is None:
            ref = find_latest_filing(ticker, as_of, forms=("10-Q",))
        if ref is None:
            return None
        if not self._is_ingested(ref.accession):
            text = fetch_filing_text(ref)
            self.ingest_text(ref, text)
        return ref

    # -- earnings call transcripts -------------------------------------------
    # Stored in the same collection with form="EARNINGS_CALL" and their own,
    # much shorter validity window (guidance is superseded quarterly).

    def transcript_ingested(self, ticker: str, year: int, quarter: int) -> bool:
        acc = f"call_{ticker.lower()}_{year}q{quarter}"
        got = self._get_col().get(where={"accession": acc}, limit=1)
        return bool(got["ids"])

    def ingest_transcript(
        self, ticker: str, year: int, quarter: int, event_date: str, text: str
    ) -> int:
        """Chunk and index one earnings call transcript. Returns chunk count."""
        from tradingagents.dataflows.filings import FilingRef
        ref = FilingRef(
            ticker=ticker.upper(), cik=0, form="EARNINGS_CALL",
            accession=f"call_{ticker.lower()}_{year}q{quarter}",
            filed_date=event_date, primary_doc="",
        )
        return self.ingest_text(ref, text)

    def ensure_transcript(self, ticker: str, as_of: str):
        """Fetch+index the latest call on or before as_of. Returns TranscriptRef|None."""
        from tradingagents.dataflows.transcripts import fetch_latest_transcript
        got = fetch_latest_transcript(ticker, as_of)
        if got is None:
            return None
        ref, text = got
        if not self.transcript_ingested(ticker, ref.year, ref.quarter):
            self.ingest_transcript(ticker, ref.year, ref.quarter, ref.event_date, text)
        return ref

    def search_transcripts(
        self, ticker: str, query: str, as_of: str, n_results: int = 4
    ) -> list[dict]:
        """Semantic search over earnings call chunks (own validity window)."""
        col = self._get_col()
        ef = get_shared_embedding_fn()
        oldest = datetime.fromisoformat(as_of) - timedelta(
            days=self._transcript_validity_days
        )
        results = col.query(
            query_embeddings=[ef.encode_query(query)],
            n_results=n_results,
            where={"$and": [
                {"wing": {"$eq": ticker.lower()}},
                {"form": {"$eq": "EARNINGS_CALL"}},
                {"filed_date": {"$lte": _date_to_int(as_of)}},
                {"filed_date": {"$gte": _date_to_int(oldest.strftime("%Y-%m-%d"))}},
            ]},
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "accession": meta.get("accession"),
                "form": meta.get("form"),
                "filed_date": meta.get("filed_date"),
                "chunk_index": meta.get("chunk_index"),
                "similarity": round(1.0 - dist, 3),
            })
        return hits

    # -- retrieval -----------------------------------------------------------

    def search(
        self,
        ticker: str,
        query: str,
        as_of: str,
        n_results: int = 4,
    ) -> list[dict]:
        """Semantic search over this ticker's filing chunks.

        Both bounds are enforced INSIDE the vector search (chroma `where`
        pre-filter), not by post-hoc discarding:
          filed_date <= as_of                      (causal isolation)
          filed_date >= as_of - validity_days     (staleness window)
        """
        col = self._get_col()
        ef = get_shared_embedding_fn()
        oldest = datetime.fromisoformat(as_of) - timedelta(days=self._validity_days)
        results = col.query(
            query_embeddings=[ef.encode_query(query)],
            n_results=n_results,
            where={"$and": [
                {"wing": {"$eq": ticker.lower()}},
                # Transcripts share the collection but have their own search
                # path and validity window — exclude them here.
                {"form": {"$in": ["10-K", "10-Q"]}},
                {"filed_date": {"$lte": _date_to_int(as_of)}},
                {"filed_date": {"$gte": _date_to_int(oldest.strftime("%Y-%m-%d"))}},
            ]},
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "accession": meta.get("accession"),
                "form": meta.get("form"),
                "filed_date": meta.get("filed_date"),
                "chunk_index": meta.get("chunk_index"),
                "similarity": round(1.0 - dist, 3),
            })
        return hits


_store: Optional[FilingStore] = None
_store_lock = threading.Lock()


def get_filing_store() -> FilingStore:
    """Return the process-wide FilingStore (path from config)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                from tradingagents.dataflows.config import get_config
                cfg = get_config()
                _store = FilingStore(
                    index_path=cfg.get("filings_index_dir", "./data/filings_index"),
                    validity_days=cfg.get("filing_validity_days", _DEFAULT_VALIDITY_DAYS),
                    transcript_validity_days=cfg.get(
                        "transcript_validity_days", _DEFAULT_TRANSCRIPT_VALIDITY_DAYS
                    ),
                )
    return _store
