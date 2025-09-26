"""Environment helpers for demos.

By default, we load .env and disable LangSmith tracing to avoid
noisy connection errors in environments without access/proxy setup.
Set `ALLOW_LANGSMITH=1` to explicitly enable LangSmith for tracing.
"""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv


_TRUTHY = {"1", "true", "True", "yes", "on", "ON"}


def _set_env_bulk(pairs: Iterable[tuple[str, str]]) -> None:
    for k, v in pairs:
        os.environ[k] = v


def init_env() -> None:
    """Load .env and disable LangSmith tracing unless explicitly allowed.

    This prevents background attempts to reach LangSmith when running
    local demos (e.g., shell tool), especially in proxied/offline envs.
    """
    load_dotenv()

    if os.getenv("ALLOW_LANGSMITH") in _TRUTHY:
        return

    # Force-disable tracing flags; keep keys intact so users can enable
    # later by exporting ALLOW_LANGSMITH=1 for the same environment.
    _set_env_bulk(
        [
            ("LANGCHAIN_TRACING_V2", "false"),
            ("LANGSMITH_TRACING", "false"),
        ]
    )

