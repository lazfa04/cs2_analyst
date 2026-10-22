"""MCP protocol layer: thin tool wrappers over the demo store, analysis and API clients.

Nothing in here computes anything. Each tool resolves its arguments, calls a
pure function, and returns a dict. Keeping this layer thin means the same
logic is reused by tests, the agent and the evals without going through MCP.

Tool docstrings are prompts: the model chooses tools from the name, the
docstring and the parameter descriptions, so they say WHEN to use a tool and
what to call first, not just what it returns.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import analysis
from .clients import FaceitClient, LeetifyClient
from .config import Settings, get_settings
from .demo_store import DemoStore
from .errors import AnalystError, DemoNotFound

INSTRUCTIONS = """\
Counter-Strike 2 analysis tools over parsed match demos plus FACEIT and Leetify stats.
Workflow: list_demos (or ingest_demo with a .dem path) -> get_match_overview -> drill down with
get_round_summary, get_opening_duels, get_utility_stats and find_kills.
Every event in a result carries `round` and `tick`; cite them when you make a claim about the match.
Sides are always "CT" or "T". Teams are labelled by starting side because sides swap at halftime.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
READ_ONLY_REMOTE = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)

DemoId = Annotated[str, Field(description="12-character id returned by ingest_demo or list_demos.")]
Player = Annotated[str, Field(description="Player name as shown in the scoreboard (partial names work) or a Steam64 ID.")]


def guided(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn our errors into ToolError so the model sees the recovery hint.

    Without this, the MCP SDK replaces any unexpected exception with a generic
    "Error executing tool", and the model can't self-correct.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except AnalystError as e:
            raise ToolError(str(e)) from e

    return wrapper


def build_server(
    settings: Settings | None = None,
    store: DemoStore | None = None,
    faceit: Callable[[], FaceitClient] | None = None,
    leetify: Callable[[], LeetifyClient] | None = None,
) -> MCPServer:
    settings = settings or get_settings()
    store = store or DemoStore(settings.cache_dir)
    # Clients are built lazily so a missing FACEIT key only breaks FACEIT tools.
    faceit = faceit or functools.cache(lambda: FaceitClient(settings.faceit_api_key, settings.http_cache_ttl_s))
    leetify = leetify or functools.cache(lambda: LeetifyClient(settings.leetify_api_key, settings.http_cache_ttl_s))

    mcp = MCPServer("cs2-analyst", instructions=INSTRUCTIONS, version="0.1.0")

    def table(demo_id: str, name: str):
        return store.table(demo_id, name)

    def tick_rate(demo_id: str) -> float:
        return store.meta(demo_id)["tick_rate"]

    def steamid(demo_id: str, player: str) -> int:
        return analysis.resolve_player(table(demo_id, "players"), player)["steamid"]

    # ------------------------------------------------------------ demo tools

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def list_demos() -> dict[str, Any]:
        """List demos that are already parsed and ready to analyse. Call this before asking the user for a file."""
        demos = store.list_demos()
        return {"count": len(demos), "demos": demos}

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, openWorldHint=False))
    @guided
    def ingest_demo(
        path: Annotated[str, Field(description="Path to a CS2 .dem file, or a bare file name inside CS2_DEMO_DIR.")],
        force: Annotated[bool, Field(description="Re-parse even if this demo is already cached.")] = False,
    ) -> dict[str, Any]:
        """Parse a CS2 demo once and cache it. Returns the demo_id every other demo tool needs.

        Safe to call twice: an already-parsed demo returns instantly from cache.
        """
        p = Path(path).expanduser()
        relative = not p.is_absolute()
        if relative and settings.demo_dir and not p.exists():
            p = settings.demo_dir / p
        if not p.is_file() and relative:
            # Say exactly where we looked: the usual cause is CS2_DEMO_DIR being unset or wrong.
            where = (f"in CS2_DEMO_DIR ({settings.demo_dir})" if settings.demo_dir
                     else f"in the current folder ({Path.cwd()}), because CS2_DEMO_DIR is not set")
            raise DemoNotFound(f"No file '{path}' {where}. Pass the full path to the .dem file, or set CS2_DEMO_DIR.")
        return store.ingest(p, force=force)

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_match_overview(demo_id: DemoId) -> dict[str, Any]:
        """Map, final score per team (with CT/T split) and the full scoreboard: rating (an approximation of
        HLTV Rating 2.0, 1.00 is average) with each player's rank in their team and in the match, impact,
        K/D/A, ADR, KAST, HS%, opening kills/deaths, traded deaths, clutches and utility damage.
        Rank and compare players by rating, not kills. Start every match analysis here."""
        return analysis.match_overview(
            store.meta(demo_id), table(demo_id, "rounds"), table(demo_id, "stats"),
            table(demo_id, "players"), table(demo_id, "round_players"),
        )

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_round_summary(
        demo_id: DemoId,
        round: Annotated[int, Field(ge=1, description="Round number, starting at 1.")],
    ) -> dict[str, Any]:
        """Everything about one round: winner and reason, each side's buy type, every player's money and
        loadout at freeze end, the kill feed with timings, and bomb plant/defuse. Use it to explain why
        a specific round was won or lost."""
        return analysis.round_summary(
            round, table(demo_id, "rounds"), table(demo_id, "kills"), table(demo_id, "round_players"),
            table(demo_id, "round_economy"), table(demo_id, "bomb"), tick_rate(demo_id),
        )

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_opening_duels(
        demo_id: DemoId,
        player: Player,
        limit: Annotated[int, Field(ge=1, le=30, description="Max duels to list.")] = 10,
    ) -> dict[str, Any]:
        """Opening duels (the first kill of a round) a player took part in: won/lost overall and per side,
        how often the round was won afterwards, and each duel with opponent, weapon, timing, position and
        whether a teammate traded the death. Good first check for entry players and AWPers."""
        return analysis.opening_duels(
            table(demo_id, "kills"), table(demo_id, "rounds"), steamid(demo_id, player), tick_rate(demo_id), limit
        )

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_utility_stats(demo_id: DemoId, player: Player) -> dict[str, Any]:
        """A player's utility usage: flashes thrown, enemies blinded and for how long, team-flashes and
        self-flashes (with the worst ones by round and tick), and grenade damage by type. Team and self
        flashes only count blinds of 1.5s or more; shorter ones are reported as minor."""
        sid = steamid(demo_id, player)
        return analysis.utility_summary(table(demo_id, "stats"), table(demo_id, "blinds"), table(demo_id, "damages"), sid)

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def find_kills(
        demo_id: DemoId,
        player: Annotated[str | None, Field(description="Only kills involving this player.")] = None,
        role: Annotated[Literal["attacker", "victim", "any"], Field(
            description="With `player`: their kills (attacker), their deaths (victim), or both.")] = "any",
        round: Annotated[int | None, Field(ge=1, description="Only this round.")] = None,
        side: Annotated[Literal["CT", "T"] | None, Field(description="Only kills made BY this side.")] = None,
        weapon: Annotated[str | None, Field(description="Substring of the weapon name, e.g. 'awp', 'ak47'.")] = None,
        headshot: Annotated[bool | None, Field(description="Only headshots (true) or only body shots (false).")] = None,
        traded: Annotated[bool | None, Field(
            description="Was the victim's death traded by a teammate within 5.25s? false finds untraded deaths.")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
        offset: Annotated[int, Field(ge=0, description="Pagination: pass next_offset from the previous call.")] = 0,
    ) -> dict[str, Any]:
        """Search the kill feed with filters and pagination. Exit frags after the round was decided are
        excluded. Example: a player's untraded deaths = player=<name>, role=victim, traded=false."""
        sid = steamid(demo_id, player) if player else None
        return analysis.find_kills(
            table(demo_id, "kills"), steamid=sid, role=role, round_num=round, side=side, weapon=weapon,
            headshot=headshot, traded=traded, limit=limit, offset=offset,
        )

    # ------------------------------------------------------------ stats APIs

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    @guided
    def get_faceit_profile(nickname: Annotated[str, Field(description="Exact FACEIT nickname (case-sensitive).")]) -> dict[str, Any]:
        """FACEIT level, ELO and lifetime CS2 stats for a player. Also returns their Steam64 ID."""
        return faceit().profile(nickname)

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    @guided
    def get_recent_faceit_matches(
        nickname: Annotated[str, Field(description="Exact FACEIT nickname (case-sensitive).")],
        limit: Annotated[int, Field(ge=1, le=30)] = 10,
    ) -> dict[str, Any]:
        """A player's most recent FACEIT CS2 matches: map, result, score, K/D, ADR and HS%. Use it to spot
        form trends or pick which match to analyse."""
        return faceit().recent_matches(nickname, limit)

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    @guided
    def get_leetify_profile(
        steam64_id: Annotated[str, Field(description="17-digit Steam64 ID.")],
    ) -> dict[str, Any]:
        """Leetify ratings (aim, positioning, utility, clutch, opening), ranks across platforms, key skill
        stats like reaction time and trade success, and recent matches. Aggregate stats, not demo-level."""
        return leetify().profile(steam64_id)

    return mcp


def main() -> None:
    build_server().run()  # stdio transport: what Claude Desktop and the MCP Inspector expect


if __name__ == "__main__":
    main()
