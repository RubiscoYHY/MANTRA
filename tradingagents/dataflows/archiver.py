"""Raw-post archival interface.

Local machines have no room for a growing archive of raw social posts, so the
default implementation is a no-op. The collection pipeline calls
``get_archiver().archive(...)`` unconditionally — enabling persistence on a
cloud deployment is a one-line config change (``archive_raw_social: True``),
no business-code edits.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class RawPostArchiver(ABC):
    """Sink for raw fetched posts/articles before any filtering."""

    @abstractmethod
    def archive(self, posts: list[dict], source: str, fetched_at: str) -> None:
        """Persist a batch of raw post dicts.

        Args:
            posts:      Raw post/article dicts exactly as fetched.
            source:     Origin label, e.g. "stocktwits", "reddit".
            fetched_at: ISO date string of the fetch (used for partitioning).
        """


class NullArchiver(RawPostArchiver):
    """Default: discard everything (local mode, no bulk storage)."""

    def archive(self, posts: list[dict], source: str, fetched_at: str) -> None:
        return None


class JsonlDiskArchiver(RawPostArchiver):
    """Append raw posts to ``{root}/{source}/{date}.jsonl.zst`` (or plain
    ``.jsonl`` when zstandard is not installed). For cloud deployments."""

    def __init__(self, root: str = "./data/raw_social"):
        self._root = Path(root)

    def archive(self, posts: list[dict], source: str, fetched_at: str) -> None:
        if not posts:
            return
        day = fetched_at[:10]
        target_dir = self._root / source
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(p, default=str) + "\n" for p in posts)
        try:
            import zstandard as zstd
            path = target_dir / f"{day}.jsonl.zst"
            cctx = zstd.ZstdCompressor()
            with open(path, "ab") as f:
                f.write(cctx.compress(payload.encode("utf-8")))
        except ImportError:
            path = target_dir / f"{day}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(payload)
        logger.debug("Archived %d posts to %s", len(posts), path)


_archiver: RawPostArchiver | None = None


def get_archiver() -> RawPostArchiver:
    """Return the configured archiver (NullArchiver unless archive_raw_social)."""
    global _archiver
    if _archiver is None:
        from .config import get_config
        cfg = get_config()
        if cfg.get("archive_raw_social", False):
            _archiver = JsonlDiskArchiver(cfg.get("raw_social_archive_dir", "./data/raw_social"))
        else:
            _archiver = NullArchiver()
    return _archiver
