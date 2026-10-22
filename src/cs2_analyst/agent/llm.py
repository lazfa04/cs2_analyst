"""Model access behind a tiny interface, so tests can swap in a scripted fake.

The loop only needs: send system + tools + messages, get back content blocks,
token usage and a stop reason. `raw_content` is passed back verbatim as the
assistant turn, which keeps any thinking blocks intact between tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


@dataclass
class Block:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass
class LLMResponse:
    blocks: list[Block]
    raw_content: Any
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None

    @property
    def tool_uses(self) -> list[Block]:
        return [b for b in self.blocks if b.type == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.type == "text" and b.text).strip()


class LLM(Protocol):
    async def create(self, *, model: str, system: str, tools: list[dict], messages: list[dict],
                     max_tokens: int) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, api_key: str | None = None):
        from anthropic import AsyncAnthropic  # imported here so tests never need the SDK configured

        # The SDK already retries 429/5xx with backoff; 4 retries rides out short rate limits.
        self._client = AsyncAnthropic(api_key=api_key, max_retries=4)

    async def create(self, *, model: str, system: str, tools: list[dict], messages: list[dict],
                     max_tokens: int) -> LLMResponse:
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            # Caches the whole prefix (system, tools, history) at the newest block, so each loop
            # step only pays full price for what's new since the previous step.
            cache_control={"type": "ephemeral"},
        )
        blocks = [
            Block(type=b.type, text=getattr(b, "text", None), id=getattr(b, "id", None),
                  name=getattr(b, "name", None), input=getattr(b, "input", None))
            for b in resp.content
        ]
        usage = {k: int(getattr(resp.usage, k, 0) or 0) for k in USAGE_KEYS}
        return LLMResponse(blocks=blocks, raw_content=resp.content, usage=usage, stop_reason=resp.stop_reason)
