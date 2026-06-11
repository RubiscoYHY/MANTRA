# Concurrency-Safe Architecture for 10-K/10-Q Integration

## Background

Fundamental Analyst will gain 10-K/10-Q analysis capability. This involves:
- BERT-based labeling of 10-K/10-Q sections (large text, many chunks)
- Embedding and storing labeled chunks into ChromaDB
- Semantic retrieval during analysis

Meanwhile, Social Analyst uses FinBERT for sentiment analysis and BGE for
memory retrieval. When `parallel_analysts` is enabled, these run concurrently
via `ThreadPoolExecutor`. Three concurrency hazards arise:

1. **ChromaDB (SQLite) concurrent access** — write-write and read-write conflicts
2. **BERT model concurrent inference** — PyTorch forward() is not thread-safe
3. **Long-running 10-K ingestion blocking other analysts** — upsert of hundreds
   of chunks holds resources

---

## Proposed Architecture

### 1. Centralized I/O Lock in TradingMemoryStore

Add a `threading.RLock` that serializes all ChromaDB read/write operations.

```python
class TradingMemoryStore:
    def __init__(self, ...):
        ...
        self._io_lock = threading.RLock()

    def _store_text(self, ...):
        with self._io_lock:
            col = self._get_col()
            col.upsert(...)

    def _search_text(self, ...):
        with self._io_lock:
            col = self._get_col()
            query_vec = self._ef.encode_query(query)  # model inference also inside lock
            results = col.query(...)
```

**Trade-off**: Simple and correct, but serializes everything. Acceptable because
ChromaDB operations are fast (~ms) relative to LLM calls (~seconds). The lock
is never held during LLM inference, so analysts still run in parallel where it
matters most.

**Alternative (future)**: Replace `RLock` with `ReadWriteLock` if profiling shows
read contention is a bottleneck. Unlikely given current usage patterns.

### 2. Thread-Safe Model Singleton

BGE and FinBERT models should each be loaded exactly once and protected during
inference. Two options:

#### Option A: Lock per model (recommended for now)

```python
class _BGEEmbeddingFunction:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, device=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(device=device)
        return cls._instance

    def __init__(self, device=None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("BAAI/bge-base-en-v1.5", ...)
        self._infer_lock = threading.Lock()

    def encode_query(self, query: str) -> list[float]:
        with self._infer_lock:
            vec = self._model.encode(...)
        return vec.tolist()

    def __call__(self, input):
        with self._infer_lock:
            vecs = self._model.encode(list(input), ...)
        return vecs.tolist()
```

Apply the same pattern to FinBERT (`_get_finbert_pipeline()`).

#### Option B: Dedicated inference thread (future, if needed)

Route all model inference through a single-threaded queue. Overkill for now,
but useful if we add more models (e.g., a 10-K section classifier).

### 3. Batched 10-K/10-Q Ingestion

A single 10-K filing can be 50,000+ words. Naive per-chunk upsert would hold
the I/O lock for minutes. Instead:

```python
def store_filing_chunks(self, ticker: str, filing_type: str,
                        chunks: list[dict], batch_size: int = 64) -> int:
    """
    Ingest pre-chunked 10-K/10-Q sections into ChromaDB.
    Releases the I/O lock between batches so other analysts can query.

    Each chunk dict: {text: str, section: str, page: int}
    """
    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        docs = [c["text"] for c in batch]
        ids = [self._filing_chunk_id(ticker, filing_type, i + j) for j, c in enumerate(batch)]
        metas = [self._filing_metadata(ticker, filing_type, c) for c in batch]
        with self._io_lock:
            col = self._get_col()
            col.upsert(documents=docs, ids=ids, metadatas=metas)
        stored += len(batch)
    return stored
```

Key design decisions:
- **batch_size=64**: ChromaDB handles batch upserts efficiently; 64 is a sweet
  spot between lock-hold duration and overhead.
- **Lock released between batches**: Other analysts can interleave queries.
- **Pre-chunked input**: Chunking strategy (by section, by token window, etc.)
  is the caller's responsibility, not the memory store's.

### 4. 10-K/10-Q Room Schema

```python
_EXPIRY_DAYS: dict[str, Optional[int]] = {
    ...
    "filing_10k":  365,   # annual filing, valid for ~1 year
    "filing_10q":  120,   # quarterly filing, valid for ~1 quarter
}
```

Metadata per chunk:
```python
{
    "wing":         ticker,
    "room":         "filing_10k" | "filing_10q",
    "section":      "risk_factors" | "mda" | "financial_statements" | ...,
    "filing_date":  "2025-02-15",      # actual SEC filing date
    "fiscal_period": "2024-Q4",        # fiscal period covered
    "chunk_index":  0,                 # position within section
    "valid_from":   filing_date_int,   # causal isolation: only visible after filing
    "expires_at":   expiry_date_int,
}
```

### 5. Filing Retrieval API

```python
def retrieve_filing_context(
    self, ticker: str, filing_type: str, query: str,
    sections: list[str] | None = None, n_results: int = 5,
) -> list[dict]:
    """
    Semantic search over 10-K/10-Q chunks for a ticker.

    Args:
        filing_type: "10k" or "10q"
        sections:    Optional filter, e.g. ["risk_factors", "mda"]
        query:       Current analysis context as search query
    """
```

Fundamental Analyst calls this during analysis. The query is the current market
situation text (same pattern as `retrieve_reflections()`).

---

## Execution Order

1. **Phase 1 (now)**: The `_init_lock` fix already committed handles the
   immediate crash. No further changes needed until 10-K work begins.

2. **Phase 2 (10-K MVP)**:
   - Add `_io_lock` to `_store_text` / `_search_text`
   - Add `_infer_lock` to `_BGEEmbeddingFunction`
   - Add `store_filing_chunks()` with batched upsert
   - Add `retrieve_filing_context()` with section filtering
   - Add room schema for `filing_10k` / `filing_10q`

3. **Phase 3 (optimization, if needed)**:
   - ReadWriteLock if read contention becomes measurable
   - Dedicated inference thread if model count grows
   - Async ChromaDB client if available in future chromadb versions

---

## Open Questions

- **Chunking strategy**: By SEC section headers? Fixed token window with
  overlap? Hybrid (section-aware windows)? Depends on how structured the
  parsed 10-K text is.
- **Filing source**: SEC EDGAR API? Pre-downloaded PDFs? This affects the
  ingestion pipeline upstream of the memory store.
- **BERT labeler**: What labels? Section classification? Sentiment per section?
  Risk factor extraction? This determines whether we need a separate fine-tuned
  model or can reuse FinBERT.
- **Deduplication**: If the same 10-K is ingested twice (e.g., backtest restart),
  the deterministic ID scheme handles it via upsert. But we should verify the
  ID scheme is stable across runs.
