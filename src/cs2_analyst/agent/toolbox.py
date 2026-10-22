"""The agent reaches the analysis tools through MCP, as a client of our own server.

Going through the protocol (instead of importing analysis.py directly) means
the agent uses exactly what Claude Desktop uses: same tools, same
descriptions, same error messages. The server runs in-process here, so there
is no subprocess or network cost.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import Client

# Setup is done by the CLI before the loop starts, so the agent doesn't need these.
SETUP_ONLY = {"ingest_demo", "list_demos"}


class Toolbox:
    def __init__(self, client: Client):
        self._c = client

    @classmethod
    @asynccontextmanager
    async def connect(cls, server: Any) -> AsyncIterator["Toolbox"]:
        async with Client(server) as client:
            yield cls(client)

    async def schemas(self, exclude: set[str] = SETUP_ONLY) -> list[dict[str, Any]]:
        tools = (await self._c.list_tools()).tools
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools if t.name not in exclude
        ]

    async def call(self, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
        """Returns (result, is_error). Results are parsed JSON; errors are the message text."""
        r = await self._c.call_tool(name, args)
        text = "\n".join(c.text for c in r.content if getattr(c, "text", None))
        if r.is_error:
            return text, True
        try:
            return json.loads(text), False
        except json.JSONDecodeError:
            return text, False
