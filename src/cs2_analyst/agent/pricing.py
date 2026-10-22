"""USD per million tokens, from https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-22).

Anthropic reports `input_tokens` WITHOUT cached tokens; cache reads and
cache writes come back as separate counters, each with its own price.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write: float  # 5-minute cache writes
    cache_read: float


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.0, 50.0, 12.5, 0.25),
    "claude-opus-5-5": Price(4.0, 20.0, 5.0, 0.20),
    "claude-sonnet-5-5": Price(2.0, 10.0, 2.5, 0.10),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.125, 0.01),
}
# Haiku 5.5 bills prompts over 100k tokens at a higher tier.
HAIKU_LONG = Price(0.50, 2.50, 0.625, 0.05)


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None  # unknown model: report tokens, don't guess a price
    if model == "claude-haiku-5-5" and input_tokens + cache_read + cache_write > 100_000:
        price = HAIKU_LONG
    total = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read * price.cache_read
        + cache_write * price.cache_write
    )
    return round(total / 1_000_000, 6)
