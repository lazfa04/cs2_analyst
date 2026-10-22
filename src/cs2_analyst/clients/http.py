"""Shared HTTP plumbing: TTL cache + retry with backoff.

Why cache: an agent often asks the same thing twice in one investigation
(profile, then recent matches, then profile again). A short TTL saves quota
and latency without serving stale data for long.
Why retry: third-party APIs rate-limit (429) and blip (5xx). Retrying a
read-only GET is always safe because it's idempotent.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..errors import ApiError

RETRY_STATUS = {429, 500, 502, 503, 504}


class CachedClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        ttl_s: int = 300,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        service: str = "API",
    ):
        self._http = httpx.Client(base_url=base_url, headers=headers or {}, timeout=15.0, transport=transport)
        self._ttl = ttl_s
        self._max_retries = max_retries
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self.service = service

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        key = (path, tuple(sorted((params or {}).items())))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]

        delay = 0.5
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise ApiError(f"{self.service} is unreachable ({e.__class__.__name__}). Try again later.") from e
            else:
                if resp.status_code not in RETRY_STATUS or attempt == self._max_retries:
                    break
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
            time.sleep(delay)
            delay *= 2

        if resp.status_code == 429:
            raise ApiError(f"{self.service} rate limit hit. Wait a minute before calling it again.", 429)
        if resp.status_code in (401, 403):
            raise ApiError(f"{self.service} rejected the API key (HTTP {resp.status_code}). Check the key in .env.",
                           resp.status_code)
        if resp.status_code >= 400:
            raise ApiError(self._error_message(resp), resp.status_code)
        data = resp.json()
        self._cache[key] = (time.monotonic(), data)
        return data

    def _error_message(self, resp: httpx.Response) -> str:
        return f"{self.service} returned HTTP {resp.status_code} for {resp.request.url.path}."
