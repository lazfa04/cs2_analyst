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
get_round_list, get_round_summary, get_opening_duels, get_utility_stats, get_clutches and find_kills.
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
    log_level: str = "INFO",
) -> MCPServer:
    settings = settings or get_settings()
    store = store or DemoStore(settings.cache_dir)
    # Clients are built lazily so a missing FACEIT key only breaks FACEIT tools.
    faceit = faceit or functools.cache(lambda: FaceitClient(settings.faceit_api_key, settings.http_cache_ttl_s))
    leetify = leetify or functools.cache(lambda: LeetifyClient(settings.leetify_api_key, settings.http_cache_ttl_s))

    mcp = MCPServer("cs2-analyst", instructions=INSTRUCTIONS, version="0.1.0", log_level=log_level)

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
        K/D/A, ADR, KAST, HS%, opening kills/deaths, traded deaths, clutches, 3k/4k/aces and utility damage.
        Rank and compare players by rating, not kills. Start every match analysis here."""
        return analysis.match_overview(
            store.meta(demo_id), table(demo_id, "rounds"), table(demo_id, "stats"),
            table(demo_id, "players"), table(demo_id, "round_players"), kills=table(demo_id, "kills"),
            damages=table(demo_id, "damages"), bomb=table(demo_id, "bomb"),
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
        """A player's utility usage. Grenades thrown by type (smokes, flashes, HE, molotovs), split by side
        and by before/after the bomb plant (T before plant = executes, CT after plant = retakes) and compared
        with teammates. Flashes: enemies blinded and for how long, team and self flashes (1.5s or more;
        shorter ones are minor), and which team flashes were costly (the teammate died while blind).
        Grenade damage by type. Use it for any question about utility or a support role."""
        sid = steamid(demo_id, player)
        flashes = analysis.utility_summary(
            table(demo_id, "stats"), table(demo_id, "blinds"), table(demo_id, "damages"), sid,
            kills=table(demo_id, "kills"), tick_rate=tick_rate(demo_id),
        )
        thrown = analysis.grenade_usage(
            table(demo_id, "grenade_throws"), table(demo_id, "bomb"), table(demo_id, "round_players"),
            table(demo_id, "players"), sid, flashes.get("rounds_played") or 0,
        )
        return {**flashes, "grenades": thrown}

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_multikills(
        demo_id: DemoId,
        player: Annotated[str | None, Field(description="Only this player's multi-kills. Omit for everyone's.")] = None,
        min_kills: Annotated[int, Field(ge=2, le=5, description="Only rounds with at least this many kills (5 = aces).")] = 2,
    ) -> dict[str, Any]:
        """Multi-kill rounds: 2k, 3k, 4k and aces per player with the round numbers, plus each multi-kill
        round's victims, weapons, ticks and whether the round was won. Use it for "when did I get the ace",
        "how many 4ks", or to find a player's big rounds."""
        sid = steamid(demo_id, player) if player else None
        return analysis.multikills(table(demo_id, "kills"), table(demo_id, "rounds"), sid, min_kills)

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_clutches(
        demo_id: DemoId,
        player: Annotated[str | None, Field(description="Only this player's clutches. Omit for everyone's.")] = None,
    ) -> dict[str, Any]:
        """Clutches the way players count them: every round a player won as the last one alive (any odds) and
        every real attempt they lost (any 1v1; a 1v2/1v3 where they got a kill, did damage, planted or went for
        the defuse; a 1v4/1v5 where they got a kill or planted). Saves and hopeless or unattempted last-alive rounds are listed apart as not_clutches.
        Each with round, situation (1v1, 1v2, ...), kills, outcome and survival; totals per situation and
        per player. The scoreboard's clutch columns use the same rules."""
        sid = steamid(demo_id, player) if player else None
        return analysis.clutches(table(demo_id, "kills"), table(demo_id, "rounds"), table(demo_id, "round_players"), sid,
                                 damages=table(demo_id, "damages"), bomb=table(demo_id, "bomb"))

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_round_list(demo_id: DemoId) -> dict[str, Any]:
        """The whole match round by round in one compact table: winner, side, how it ended, which team got
        the first kill, bomb plant site, each team's buy (pistol/eco/force/full) and the running score.
        Also halves, each team's rounds won per buy type and first-kill conversion. Use it to explain
        match flow, momentum or economy, instead of calling get_round_summary for many rounds."""
        return analysis.round_list(
            table(demo_id, "rounds"), table(demo_id, "kills"), table(demo_id, "round_players"),
            table(demo_id, "players"), table(demo_id, "round_economy"), table(demo_id, "bomb"),
            max_rounds=store.meta(demo_id).get("max_rounds"),
        )

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
        excluded. Besides the page of kills, it returns counts over ALL matches: total, the list of rounds,
        a CT/T split, counts per attacker and per victim, and how many were openings, trades or in rounds the
        killer's team won. Example: a player's untraded deaths = player=<name>, role=victim, traded=false;
        who killed them most is by_attacker."""
        sid = steamid(demo_id, player) if player else None
        return analysis.find_kills(
            table(demo_id, "kills"), rounds=table(demo_id, "rounds"), steamid=sid, role=role, round_num=round, side=side, weapon=weapon,
            headshot=headshot, traded=traded, limit=limit, offset=offset,
        )

    # ------------------------------------------------------------ win probability (phase 4)

    @functools.cache
    def winprob_model():
        from .winprob import model as wp

        return wp.load(settings.cache_dir / "winprob")[0]  # ModelMissing is an AnalystError: guided shows it

    def swing_states(demo_id: str):
        from .winprob import swing as sw
        from .winprob.features import states_for

        return sw.with_probabilities(winprob_model(), states_for(store, demo_id))

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_death_review(
        demo_id: DemoId,
        player: Player,
        round: Annotated[int | None, Field(ge=1, description="Only this round's deaths.")] = None,
    ) -> dict[str, Any]:
        """A player's deaths with the context that explains them: walking distance to the nearest living
        teammate and how many could trade, whether it was traded, opening deaths with or without a teammate's
        flash, whether they died blind, grenades still held, and how much each death cost the round (swing).
        Flags isolated deaths, dry opening deaths, dying with utility and dying blind; `headline` is the flag that
        says why a death happened and `flag_points_at` whose it was to fix (player or teammates). A lurker's isolated
        deaths while lurking are not flagged (see spacing_profile, isolated_while_lurking). Use it for "what went
        wrong", positioning, trading and mistakes questions. Flags are facts to weigh, not verdicts."""
        from . import coach

        try:
            model = winprob_model()
        except AnalystError:
            model = None
        out = coach.player_review(store, demo_id, steamid(demo_id, player), round_num=round, model=model)
        rows = out.pop("deaths_detail")
        out["deaths_detail"], out["truncated"] = rows[:40], len(rows) > 40
        return out

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_win_probability(
        demo_id: DemoId,
        round: Annotated[int, Field(ge=1, description="Round number, starting at 1.")],
    ) -> dict[str, Any]:
        """One round as a win probability timeline: T's chance to win after every kill, the plant and the
        defuse, how much each event swung it, and which factors (players alive, health, equipment, kits,
        bomb, clock) drove each estimate. Use it to find the moment a round was won or lost."""
        from .winprob import swing as sw

        states = swing_states(demo_id)
        if round not in set(states["round"].to_list()):
            valid = sorted(set(states["round"].to_list()))
            raise AnalystError(f"Round {round} has no states. Valid rounds: {valid[0]}-{valid[-1]}.")
        return sw.round_timeline(states, winprob_model(), round, tick_rate(demo_id))

    @mcp.tool(annotations=READ_ONLY)
    @guided
    def get_swing_ratings(demo_id: DemoId) -> dict[str, Any]:
        """Round swing per player: the average change in their team's chance to win the round caused by their
        kills, assists, deaths, plants and defuses (percentage points per round), next to their rating, with
        each player's biggest plays and costliest moments. Rating weighs every kill the same; swing shows which
        kills mattered (opening kills, clutch and post-plant kills). Use it for impact questions."""
        from .winprob import swing as sw

        rp = table(demo_id, "round_players")
        labels = analysis.team_labels(rp, table(demo_id, "players"))
        ratings = dict(analysis.with_rating(table(demo_id, "stats")).select("steamid", "rating").iter_rows())
        return sw.swing_ratings(swing_states(demo_id), rp, labels, ratings)

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
