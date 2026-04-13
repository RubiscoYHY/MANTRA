# Third-Party License Notices

TradingAgents is licensed under the **Apache License 2.0**. See `LICENSE`.

The following third-party components are used under their own licenses. Copies of each license are available at the paths noted below (some require cloning the respective repository first).

---

## MemPalace

| Field | Detail |
|-------|--------|
| **Purpose** | Social media sentiment memory: vector retrieval (ChromaDB) + time-series knowledge graph (SQLite) |
| **Used by** | `tradingagents/agents/utils/sentiment_memory.py` (`SentimentMemoryStore`) |
| **License** | MIT — see `third_party/mempalace/LICENSE` |
| **Source** | https://github.com/milla-jovovich/mempalace — cloned to `third_party/mempalace/` |
| **Activation** | Only loaded when `use_sentiment_memory: True` in config |

> Apache 2.0 and MIT are compatible: both permit use, modification, and redistribution. The MIT license imposes no additional restrictions on this project's Apache 2.0 terms.

---

## Other Runtime Dependencies

All other dependencies (LangChain, ChromaDB, sentence-transformers, PRAW, etc.) are declared in `pyproject.toml` / `requirements.txt` and distributed under their respective licenses (Apache 2.0, MIT, BSD). Refer to each package's metadata for the full license text.
