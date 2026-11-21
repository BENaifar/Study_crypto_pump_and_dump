"""Stream the authors' raw Telegram message pickles (lists of Telethon ``to_dict()`` messages)."""

from __future__ import annotations

import pickle
from collections.abc import Iterator

from pnd_paper import settings


def iter_messages(raw_dir=settings.TELEGRAM_RAW) -> Iterator[tuple[str, object, str]]:
    """Yield (channel_id, date, text) for every non-empty channel post, one pickle at a time."""
    for path in sorted(raw_dir.glob("*.pkl")):
        with open(path, "rb") as f:
            messages = pickle.load(f)
        for m in messages:
            if not isinstance(m, dict) or m.get("_") != "Message":
                continue
            text = m.get("message")
            if text:
                yield path.stem, m.get("date"), text
        del messages
