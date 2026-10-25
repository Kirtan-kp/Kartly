"""Make untrusted values safe to store and show.

Model output, ticket text, and carrier data are all untrusted. Before such values go into the
audit log (a JSONB column on Postgres) they must be bounded in size and free of anything the
database or JSON encoder rejects: NUL characters, lone surrogates, NaN/Infinity, huge integers.
"""

from __future__ import annotations

import math
from typing import Any

MAX_STR = 200
MAX_KEY = 40
MAX_KEYS = 20
MAX_LIST = 10
MAX_DEPTH = 3
MAX_INT_BITS = 53  # larger integers are logged as a placeholder (also avoids str() limits)


def clean_str(value: str, limit: int = MAX_STR) -> str:
    """Truncate to ``limit`` characters and drop characters Postgres or UTF-8 cannot store."""
    truncated = len(value) > limit
    cleaned = value[:limit].encode("utf-8", "replace").decode("utf-8").replace("\x00", "")
    return cleaned + "…" if truncated else cleaned


def safe_json(value: Any, _depth: int = 0) -> Any:
    """Return a bounded, JSON-serialisable copy of an arbitrary value (never raises)."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= MAX_INT_BITS else "<big int>"
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return clean_str(value)
    if _depth >= MAX_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {
            clean_str(str(key), MAX_KEY): safe_json(item, _depth + 1)
            for key, item in list(value.items())[:MAX_KEYS]
        }
    if isinstance(value, list | tuple):
        return [safe_json(item, _depth + 1) for item in list(value)[:MAX_LIST]]
    return f"<{type(value).__name__}>"
