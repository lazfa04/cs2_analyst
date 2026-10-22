# cs2-analyst

An AI analyst for Counter-Strike 2 match demos. Ask it why you lost, who won the opening duels, or which deaths went untraded. It investigates the demo with a set of analysis tools, then checks every claim it makes against the data before showing you the answer.

Two parts:

1. **An MCP server** (phase 1) that turns a `.dem` file into 10 tools any MCP client can use, including Claude Desktop.
2. **A verifying agent** (phase 2) that runs its own loop over those tools, states its answer as structured claims with evidence, and has each claim checked: facts deterministically against the tool results, interpretations by a separate judge model.

```
$ uv run cs2-agent ask match.dem "How did I do in opening duels?" --player babaya0310

╭─────────────────────────── Answer ───────────────────────────╮
│ You won 6 of your 7 opening duels, including all 4 on CT.     │
╰───────────────────────────────────────────────────────────────╯
  ✓ verified   You won 6 of 7 opening duels       t2:overall.won, t2:overall.taken
  ✓ verified   All 4 of your CT-side duels were wins   t2:by_side.CT.won
Did not hold up:
  ✗ failed     You had 30 kills   (scoreboard[name=babaya0310].kills: expected '30', got 21)
4 model calls · 2 tool calls · ... · $0.01 · revisions 0
```

## Architecture

```
cs2-agent (CLI)                      Claude Desktop
      |                                    |
  agent/loop.py  model <-> tools           |
  agent/verifier.py  fact checks           |
  agent/judge.py  interpretation checks    |
      |   MCP client (in-process)          |  MCP (stdio)
      v                                    v
  server.py      protocol layer: thin tool wrappers, error translation
        |
  analysis.py    pure functions: Polars frames in, small JSON-ready dicts out
        |
  demo_store.py  parse once with awpy -> parquet cache per demo
  clients/       FACEIT + Leetify HTTP clients (TTL cache, retry with backoff)
```

The agent reaches the tools through MCP, as a client of its own server, so it uses exactly what Claude Desktop uses: same tools, descriptions and error messages. `analysis.py` has no I/O and knows nothing about MCP, so the same functions power the server, the tests, the agent and later the evals and ML pipeline.

## How the agent works

```
question ─> model ─tool_use─> run tool via MCP ─[call_id=t3] result─> model ─> ...
              └─submit_report─> verifier ─all good─> judge ─> report
                                   └─failures─> actual values sent back, agent resubmits
```

The agent can't just answer in prose. It calls `submit_report` with a summary plus a list of atomic claims:

- **Facts** carry checks: a tool call id, a path to one value in that result, and the claimed value. Example: `{"call_id": "t2", "path": "duels[round=4].traded", "expected": "true"}`. The verifier resolves the path in the recorded result and compares. No model is involved, so this step can't hallucinate.
- **Interpretations** ("you lose rounds because you peek without utility") list the facts they rest on. They count as supported only if every cited fact verified and a judge model (Haiku 5.5 by default) agrees the facts actually back the conclusion.

Failed facts go back to the agent with the real values, so it can fix or drop them (one revision by default). Whatever still fails is shown separately as "did not hold up", never mixed into the answer.

## Design decisions

### Phase 1: tools

**Parse once, query many times.** Parsing a demo takes 1 to 10 seconds. `ingest_demo` does it once and writes parquet tables keyed by a content hash of the file. Every other tool reads those small files, so an agent can ask 30 questions about one match cheaply.

**Tool outputs are summaries, not tables.** Tool results go into the model's context window, so tools return aggregates plus a capped list of the most relevant events, with `limit`, `truncated` and pagination.

**Every event carries `round` and `tick`,** so claims can cite evidence and the verifier can find it.

**Errors are written for the model.** A wrong player name returns the roster and a "did you mean". MCP SDK 2.x hides the message of any exception that isn't a `ToolError`, so `server.py` translates domain errors at the boundary. Without that the model only sees "Error executing tool" and gets stuck.

**Tool descriptions are prompts.** The model picks tools from their names and docstrings, so each one says when to use it, not just what it returns.

**Normalise once at ingest.** awpy reports sides as `counter-terrorist`, `terrorist` or team numbers depending on the table; everything becomes `CT` / `T`. Teams are labelled by starting side (or clan tag) because sides swap at halftime, so "CT" is not a team.

**Events don't come with round numbers.** awpy's kill and damage tables only have ticks, so `add_round_num` interval-joins them onto round windows. Warmup kills fall outside every window and get dropped; kills after the round was decided are flagged as exit frags.

**Rank players by rating, not kills.** The scoreboard includes a rating using the public approximation of HLTV Rating 2.0 (KAST, kills, deaths and assists per round, ADR), the same formula awpy used to ship. HLTV's current Rating 3.0 is built around round swing, how each action changed the team's chance to win the round, and its formula isn't published. Phase 4's win probability model is the route to a swing-based rating. The linear formula goes negative in a stomp, so the displayed rating is floored at 0.00 while ranking uses the raw value.

**Minor blinds aren't team flashes.** Most self-blinds in real demos are under a second (catching the edge of your own flash while turned away). Team and self flashes count only blinds of 1.5s or more; shorter ones are reported separately.

### Phase 2: agent

**Verify facts with code, not with another model.** A judge model can be wrong too. Facts are checked by resolving a path and comparing values, which is deterministic and free. The model is only used where code can't decide: whether evidence supports a conclusion.

**Precision-aware number checks.** A claim is right if the real value, rounded to the decimals the agent wrote, equals the claim: ADR 89.83 matches "90" and "89.8" but not "89". Counts must match exactly. Rounding is half-up (Python's `round()` rounds 2.5 down to 2).

**Paths with filters, not indexes.** `duels[round=4].traded` instead of `duels.3.traded`. Models miscount list positions; selecting by content is robust and reads like evidence.

**No forced tool calls.** Sonnet 5.5 and Opus 5.5 reject `tool_choice: any/tool` with a 400 error, so the loop can't force `submit_report`. It uses `strict: true` schemas (guaranteed-valid report JSON) plus prompting, a nudge if the model answers in plain text, and a final "submit now" turn at the step limit. A plain-text answer is kept but marked unverified.

**The summary can't sneak in specifics.** The judge also reviews the summary and flags any statement (a ranking, a name, a number) that no verified claim backs. This came from a real run where the summary said a player was "second on his team" without any claim behind it.

**Guardrails.** Max model turns (12), a total token budget, one revision pass, and no tool runs after the final-step message.

**Prompt caching on the whole loop.** Every step re-sends the system prompt, tools and history. A top-level `cache_control` marker caches that prefix, so each step pays full price only for what's new.

**Everything is traced.** Each run writes a JSONL file to `traces/`: every model call (tokens, cache hits, latency, cost) and every tool call (args, result, latency, errors). Costs use the published per-model prices in `agent/pricing.py`.

## Evals

"It looks right" isn't evidence, so the agent is measured on a fixed eval set: every question in `evals/cases.toml` runs against every demo (your own matches plus a public pro demo, so the numbers are reproducible).

```bash
uv run cs2-agent eval run --tag baseline        # runs every case, asks before spending
uv run cs2-agent eval report                    # metrics of the latest run
uv run cs2-agent eval label                     # label the judge's verdicts (blind)
uv run cs2-agent eval compare baseline latest   # what changed between two runs
uv run cs2-agent eval rejudge baseline          # same answers, re-judged by the current judge
```

What gets measured, and the question each metric answers:

| Metric | Question it answers |
| --- | --- |
| Facts verified / failed / unchecked | Of the facts shown to the user, how many are provably right? |
| First drafts with a wrong fact | How often the agent's first answer had a wrong fact, i.e. what you'd see with **no verifier**. The gap to "final answers with a wrong fact" is what verification buys. |
| Interpretations supported / rejected | How many conclusions survive the judge, and whether they fail on evidence (verifier) or on reasoning (judge) |
| Unbacked summary text | How often the answer slips in specifics that no claim backs |
| Judge agrees with expert | How often the judge's verdict matches a human player's, and whether it errs lenient or strict |
| Cost, latency, tool calls, cache hits | What a question costs to answer |

**Labelling.** `eval label` shows each interpretation with the verified facts behind it and asks whether the facts support it. The judge's own verdict stays hidden so it can't anchor you. Labels go to `evals/labels.jsonl` (committed, it's the human-labelled dataset) and are reused by later runs, so judge agreement is scored automatically from then on.

**Judge agreement needs a baseline.** Labels are lopsided (in the first set, 13 of 16 conclusions were fair), so a judge that rubber-stamps everything would score 81%. The report shows that trivial score next to the real one; a judge has to beat it.

**Comparing judges fairly.** `eval rejudge <run>` keeps a run's answers, resets only the judge's verdicts and asks the judge again, so two judge prompts rule on exactly the same labelled items. Judge prompts are versioned (`--judge-prompt v1|v2`); v2 was written after the first labels showed v1 rejecting fair reads 7 times out of 8 disagreements. Because v2 was shaped by those labels, its score on them is optimistic; new labels from new demos are the real test.

**Runs** are written to `evals/runs/<run>/`: `results.jsonl` (one line per case, appended as each finishes), a trace per case, and `report.md`. A run that's interrupted or hits API errors can be finished with `--resume <run>`, which only runs the missing cases. One failing case never stops the run.

**Changing something?** Run `eval run --tag <change>` and `eval compare` against the baseline. Prompt tweaks, models (`--model claude-haiku-5-5`), step limits and the judge (`--no-judge`) are all flags.

## Tools

| Tool | What it answers |
| --- | --- |
| `list_demos` | Which demos are already parsed |
| `ingest_demo` | Parse a `.dem` file, returns the `demo_id` |
| `get_match_overview` | Score per team with CT/T split, scoreboard ranked by rating (HLTV 2.0 approximation) |
| `get_round_summary` | One round: buys, every player's loadout, kill feed with timings, bomb events |
| `get_opening_duels` | A player's first-kill duels by side, round conversion, trades, positions |
| `get_utility_stats` | Flashes, enemy vs team vs self blinds (minor blinds under 1.5s separated), worst team flashes, nade damage |
| `find_kills` | Filterable, paginated kill feed (e.g. a player's untraded deaths) |
| `get_faceit_profile` | FACEIT level, ELO, lifetime stats, Steam64 ID |
| `get_recent_faceit_matches` | Recent FACEIT matches with result, K/D, ADR |
| `get_leetify_profile` | Leetify ratings, ranks and key skill stats |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY for the agent; FACEIT key optional
uv run pytest
```

### Ask the agent

```bash
uv run cs2-agent ask match.dem "Why did we lose the CT half?" --player YourName
```

`match.dem` can be a full path, a file name inside `CS2_DEMO_DIR`, or a cached demo id. Options: `--model` (default `claude-sonnet-5-5`), `--judge-model` (default `claude-haiku-5-5`), `--no-judge`, `--max-steps`, `--max-revisions`, `--json`.

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

Try the server on its own in the MCP Inspector with `uv run mcp dev src/cs2_analyst/server.py`.

Your own matchmaking demos are under `.../Counter-Strike Global Offensive/game/csgo/replays/`. FACEIT demos can be downloaded from the match room.

## Tests

`uv run pytest` runs 104 tests with no network or API key:

- a hand-scripted 2v2 match that goes through the real ingest pipeline
- mocked HTTP for the FACEIT and Leetify clients
- an in-memory MCP session for the server
- the agent loop driven by a scripted fake model: revisions, the plain-text nudge, the step limit, refused tool calls after the final step, cost accounting and trace files
- the eval runner, metrics, resume, labelling and reports, with a scripted model that gets a fact wrong and fixes it after feedback

For a check against a real match with a known result:

```bash
CS2_TEST_DEMO=/path/to/faceit-de_mirage-6de46e85.dem uv run pytest -m demo
```

That demo is public in [awpy-fixtures](https://github.com/pnxenopoulos/awpy-fixtures). The tests check the 16-14 score, FACEIT's published kills and deaths, that the opening-duel logic agrees with awpy's independent scoreboard for all 10 players, and that agent evidence paths resolve against real tool output.

## Roadmap

- [x] Phase 1: tool layer
- [x] Phase 2: agent loop with findings schema, deterministic verifier, judge and tracing
- [x] Phase 3: eval runner, metrics with a no-verifier baseline, blind labelling and judge agreement
- [ ] Phase 3b: labelled "key mistakes" per match, to measure whether the agent finds what matters
- [ ] Phase 4: explainable round win probability model (XGBoost + SHAP)
- [ ] Phase 5: web app

Built on [awpy](https://github.com/pnxenopoulos/awpy), the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and the [Anthropic API](https://platform.claude.com/docs).
