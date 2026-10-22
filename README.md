# cs2-analyst

An MCP server that turns Counter-Strike 2 match demos into tools an AI agent can use. Point Claude (or your own agent) at a `.dem` file and ask why you lost, who won the opening duels, or which deaths went untraded. Every answer comes back with round and tick references so claims can be checked against the actual match.

This is phase 1 of a larger project: the tool layer. Phase 2 is an agent loop with a claim verifier on top of these tools, and phase 3 is an eval set of human-labelled mistakes to measure it.

## Architecture

```
Claude Desktop / your agent
        |  MCP (stdio)
        v
  server.py      protocol layer: thin tool wrappers, error translation
        |
  analysis.py    pure functions: Polars frames in, small JSON-ready dicts out
        |
  demo_store.py  parse once with awpy -> parquet cache per demo
  clients/       FACEIT + Leetify HTTP clients (TTL cache, retry with backoff)
```

The layers are separate on purpose. `analysis.py` has no I/O and knows nothing about MCP, so the same functions power the server, the tests, and later the agent, evals and ML pipeline.

## Design decisions

**Parse once, query many times.** Parsing a demo takes 1 to 10 seconds. `ingest_demo` does it once and writes parquet tables keyed by a content hash of the file. Every other tool reads those small files, so an agent can ask 30 questions about one match cheaply. Re-ingesting the same file is a no-op.

**Tool outputs are summaries, not tables.** Tool results go into the model's context window. Returning thousands of raw rows costs tokens and hurts accuracy, so tools return aggregates plus a capped list of the most relevant events, with `limit`, `truncated` and pagination where needed.

**Every event carries `round` and `tick`.** That's the hook for phase 2: the agent has to cite evidence, and a verifier re-checks each claim against the data.

**Errors are written for the model.** A wrong player name returns the roster and a "did you mean". An unknown demo id lists the ingested ones. MCP SDK 2.x hides the message of any exception that isn't a `ToolError`, so `server.py` translates domain errors at the boundary. Without that the model only sees "Error executing tool" and gets stuck.

**Tool descriptions are prompts.** The model picks tools from their names and docstrings, so each one says when to use it, not just what it returns.

**Normalise once at ingest.** awpy reports sides as `counter-terrorist`, `terrorist` or team numbers depending on the table. Everything is converted to `CT` / `T` at ingest. Teams are labelled by starting side (or clan tag when the demo has one) because sides swap at halftime, so "CT" is not a team.

**Events don't come with round numbers.** awpy's kill and damage tables only have ticks, so `add_round_num` interval-joins them onto round windows. Warmup kills fall outside every window and get dropped; kills after the round was decided are flagged as exit frags and excluded from duel and kill analysis.

## Tools

| Tool | What it answers |
| --- | --- |
| `list_demos` | Which demos are already parsed |
| `ingest_demo` | Parse a `.dem` file, returns the `demo_id` |
| `get_match_overview` | Score per team with CT/T split, full scoreboard |
| `get_round_summary` | One round: buys, every player's loadout, kill feed with timings, bomb events |
| `get_opening_duels` | A player's first-kill duels by side, round conversion, trades, positions |
| `get_utility_stats` | Flashes, enemy vs team vs self blinds, worst team flashes, nade damage |
| `find_kills` | Filterable, paginated kill feed (e.g. a player's untraded deaths) |
| `get_faceit_profile` | FACEIT level, ELO, lifetime stats, Steam64 ID |
| `get_recent_faceit_matches` | Recent FACEIT matches with result, K/D, ADR |
| `get_leetify_profile` | Leetify ratings, ranks and key skill stats |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # add your FACEIT key; Leetify works without one
uv run pytest
```

Try it in the MCP Inspector:

```bash
uv run mcp dev src/cs2_analyst/server.py
```

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings > Developer > Edit Config), with your own paths:

```json
{
  "mcpServers": {
    "cs2-analyst": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/cs2-analyst", "cs2-analyst"],
      "env": { "CS2_DEMO_DIR": "/path/to/your/demos" }
    }
  }
}
```

Then ask: *"Ingest match.dem and tell me why we lost the CT half."*

Your own matchmaking demos are under `.../Counter-Strike Global Offensive/game/csgo/replays/`. FACEIT demos can be downloaded from the match room.

## Tests

`uv run pytest` runs 40 tests against a hand-scripted 2v2 match that goes through the real ingest pipeline, plus mocked HTTP for the API clients and an in-memory MCP session for the server.

For a check against a real match with a known result:

```bash
CS2_TEST_DEMO=/path/to/faceit-de_mirage-6de46e85.dem uv run pytest -m demo
```

That demo is public in [awpy-fixtures](https://github.com/pnxenopoulos/awpy-fixtures). The test checks the 16-14 score, FACEIT's published kills and deaths, and that this project's opening-duel logic agrees with awpy's independent scoreboard for all 10 players.

## Roadmap

- [x] Phase 1: tool layer (this repo)
- [ ] Phase 2: agent loop with findings schema and claim verifier
- [ ] Phase 3: eval set of labelled mistakes, recall and precision vs a no-verifier baseline
- [ ] Phase 4: explainable round win probability model (XGBoost + SHAP)

Built on [awpy](https://github.com/pnxenopoulos/awpy) and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
