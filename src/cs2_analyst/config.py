"""Runtime settings, read once from the environment (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    demo_dir: Path | None
    faceit_api_key: str | None
    leetify_api_key: str | None
    http_cache_ttl_s: int


def get_settings() -> Settings:
    demo_dir = os.getenv("CS2_DEMO_DIR")
    return Settings(
        cache_dir=Path(os.getenv("CS2_CACHE_DIR", Path.home() / ".cs2-analyst" / "cache")).expanduser(),
        demo_dir=Path(demo_dir).expanduser() if demo_dir else None,
        faceit_api_key=os.getenv("FACEIT_API_KEY") or None,
        leetify_api_key=os.getenv("LEETIFY_API_KEY") or None,
        http_cache_ttl_s=int(os.getenv("CS2_HTTP_CACHE_TTL", "300")),
    )
