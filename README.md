# cs2-analyst

An AI analyst for Counter-Strike 2 match demos. Ask it why you lost, who won the opening duels, or which deaths went untraded. It investigates the demo with a set of analysis tools, then checks every claim it makes against the data before showing you the answer.

Two parts:

1. **An MCP server** (phase 1) that turns a `.dem` file into 15 tools any MCP client can use, including Claude Desktop.
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
  agent/judge.py  conclusions + critique   |
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
              └─submit_report─> verifier ─all good─> judge ─all good─> report
                                   │                   └─issues─> critique sent back, agent resubmits
                                   └─failures─> actual values sent back, agent resubmits
```

The agent can't just answer in prose. It calls `submit_report` with a summary plus a list of atomic claims:

- **Facts** carry checks: a tool call id, a path to one value in that result, and the claimed value. Example: `{"call_id": "t2", "path": "duels[round=4].traded", "expected": "true"}`. The verifier resolves the path in the recorded result and compares. No model is involved, so this step can't hallucinate.
- **Interpretations** ("you lose rounds because you peek without utility") list the facts they rest on. They count as supported only if every cited fact verified and a judge model (Haiku 5.5 by default) agrees the facts actually back the conclusion.

Failed facts go back to the agent with the real values, so it can fix or drop them (one revision by default). Then the judge reviews the report, and anything it finds (summary text that contradicts a claim or isn't backed by one, conclusions the facts don't support) goes back once more as a critique. Whatever still fails is shown separately as "did not hold up", never mixed into the answer.

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

**Counts come from tools, never from the model.** The Falcons eval showed every submitted fact was right while half the summaries had a hand-counted mistake ("magixx killed you 4 times" when its own claims listed 5 rounds). So tools return the totals and breakdowns: `find_kills` counts all matching kills by attacker, victim, side and round (not just the page it returns), with tags for openings, trades and kills in rounds the killer's team won; `get_clutches` and `get_round_list` return per-player, per-half and per-buy-type totals.

**Grenades from fire events.** awpy's grenade and smoke tables leave the thrower empty on many demos. Every throw is also a `weapon_fire` event with the player on it, so `grenade_throws` is built from those. Flash counts match the scoreboard exactly for every player on the test demos. Throws are split by side and by before/after the bomb plant: T side before the plant is execute utility, CT side after it is the retake.

**Clutches the way players count them.** Each round is replayed from the freeze-end roster, removing players as they die, to find every moment someone is last alive with enemies up (both sides can be: a 1v3 that turns into a 1v1). awpy's scoreboard calls all of these clutches, and with post-round kills included our raw count matches it for all ten players. But that's not how players count: a 1v5 on an eco isn't a lost clutch. After labelling the Falcons eval, the rules became: any last-alive round the team **wins** is a clutch won (if he died and the bomb won it, it's flagged `won_after_death`); any lost 1v1 is a clutch lost; a lost 1v2/1v3 is a clutch lost only if he **attempted** it (got a kill, did damage, planted the bomb as T, or died near the planted bomb going for the defuse as CT); a lost 1v4/1v5 is a clutch lost only with a kill or a plant (damage alone isn't enough at those odds); surviving a lost round is a **save**; anything else is just **last alive**. Two rounds from the Falcons maps set these lines: TeSeS's eco 1v3 with no attempt (last alive) and m0NESY's 1v4 with 2 kills and a plant (a real attempt). There's no start-defuse event in the demo, so "went for the defuse" is a distance check against the plant position. On the 30-round FACEIT demo, the scoreboard's 36 "clutches" become 20. The scoreboard's clutch columns are replaced with the same rules, so tools never disagree. The replay also found one scoreboard "1v1" that only existed because of a kill after the round was over.

**Costly team flashes.** A team flash only counts as a mistake when the blinded teammate died before the blind wore off (plus 0.5s). Most team flashes cost nothing, often because the teammate walked into it.

**Overtime halves.** Teams keep their sides going into overtime, so halves split on side swaps and also at the start of each overtime period.

**Minor blinds aren't team flashes.** Most self-blinds in real demos are under a second (catching the edge of your own flash while turned away). Team and self flashes count only blinds of 1.5s or more; shorter ones are reported separately.

### Phase 2: agent

**Verify facts with code, not with another model.** A judge model can be wrong too. Facts are checked by resolving a path and comparing values, which is deterministic and free. The model is only used where code can't decide: whether evidence supports a conclusion.

**Precision-aware number checks.** A claim is right if the real value, rounded to the decimals the agent wrote, equals the claim: ADR 89.83 matches "90" and "89.8" but not "89". Counts must match exactly. Rounding is half-up (Python's `round()` rounds 2.5 down to 2).

**Paths with filters, not indexes.** `duels[round=4].traded` instead of `duels.3.traded`. Models miscount list positions; selecting by content is robust and reads like evidence.

**No forced tool calls.** Sonnet 5.5 and Opus 5.5 reject `tool_choice: any/tool` with a 400 error, so the loop can't force `submit_report`. It uses `strict: true` schemas (guaranteed-valid report JSON) plus prompting, a nudge if the model answers in plain text, and a final "submit now" turn at the step limit. A plain-text answer is kept but marked unverified.

**The summary can't sneak in specifics.** The judge also reviews the summary and flags any statement (a ranking, a name, a number) that no verified claim backs. This came from a real run where the summary said a player was "second on his team" without any claim behind it.

**The judge's findings go back to the agent.** Flagging unbacked text only tells the user; sending it back lets the agent fix it. After the verifier passes, the judge's review (contradictions, unbacked statements, rejected conclusions) is returned once as a critique, and the agent can call more tools before resubmitting. The judge's first review is kept, so the eval measures what the critique fixed. `--critic-revisions 0` turns it off.

**Contradicted is worse than unbacked.** "Unbacked" means unchecked, "contradicted" means wrong. Judge prompt v3 reports them separately, and the eval tracks contradictions as their own metric.

**Lists are checkable too.** A path can point at a list of plain values (`rounds`), and a claim like "3, 4, 6" passes only if it names exactly those values, in any order.

**Guardrails.** Max model turns (12), a total token budget, one revision pass for failed facts and one for the critique, and no tool runs after the final-step message.

**Prompt caching on the whole loop.** Every step re-sends the system prompt, tools and history. A top-level `cache_control` marker caches that prefix, so each step pays full price only for what's new.

**Everything is traced.** Each run writes a JSONL file to `traces/`: every model call (tokens, cache hits, latency, cost) and every tool call (args, result, latency, errors). Costs use the published per-model prices in `agent/pricing.py`.

### Phase 4: win probability and round swing

Rating weighs every kill the same, so it misses what the expert labels kept pointing at: TeSeS rated 0.87 on Dust2 but won two clutches and converted every opening win, and the judge kept accepting "below average" for him. Round swing (the idea behind HLTV's Rating 3.0, whose formula isn't public) measures how much each play changed the team's chance to win the round. That needs a win probability model.

**Game states.** Each round becomes a sequence of states: freeze end, then after every kill, the plant and the defuse. Features are what a caster sees on the HUD: players alive, their total health, the freeze-end buy of the players still alive, CT defuse kits, bomb down or not, round clock or bomb timer. The label is whether T won. 5 public demos give 103 rounds and 727 states; `cs2-agent model train --ingest` adds every demo in your folder.

**Evaluated leave-one-demo-out.** States from the same round are near-duplicates, so a random split would leak; every score is on a match the model never saw, against two baselines:

| Model (5 demos, 103 rounds) | Log loss | Brier | AUC |
| --- | --- | --- | --- |
| Always the T win rate | 0.724 | 0.265 | – |
| Players alive only (logistic) | 0.537 | 0.185 | 0.780 |
| Logistic, all features, signs constrained | 0.477 | 0.158 | 0.847 |
| XGBoost, monotone constraints | 0.488 | 0.163 | 0.837 |
| **Shipped: average of the two** | **0.474** | **0.157** | **0.846** |

Calibration (the thing swing depends on, since swings are differences between probabilities) is within a few points in every bucket: when it says 90%, T won 92%.

**Decisions the data forced:**
- **Monotone constraints.** With a few hundred rounds a tree model can learn that a sixth CT alive helps T. Every feature has a fixed direction (more Ts alive never lowers T's chance), enforced in XGBoost and as coefficient bounds in the logistic model. The plain logistic model had learned "more round time left hurts T", the time-pressure effect reversed by confounding; the constrained one scores better.
- **Shallow trees over accurate ones.** Deeper XGBoost had slightly better log loss but was overconfident at the extremes (said 5% when T won 14%), which inflates swings.
- **An ensemble for credit assignment.** On this little data a constrained logistic model is as good as XGBoost, and XGBoost is coarse: about 8% of kills landed in the same tree leaves before and after and got zero swing. Averaging the two halves that (4%) and cuts kills that move the probability the wrong way to under 1%. With more data XGBoost should pull ahead; the training report shows all of them, so that's visible.
- **Explanations without an extra dependency.** XGBoost's built-in TreeSHAP and the linear model's exact contributions, averaged, give the reasons behind each estimate ("CT health pushes CT, bomb planted pushes T").

**Swing credit.** For a kill, the killer's team gains what the victim's team loses: the killer gets the gain (75% with an assist, the assister 25%), the victim gets the loss. The planter gets the plant's swing, the defuser everything that was left. The starting probability (the economy) and the last jump when a round ends on time or the bomb aren't credited. A player's swing rating is their average per round, in percentage points. On the FACEIT demo, babaya0310 rates 1.02 (6th) but ranks 3rd on swing; oh-_- rates 1.18 (3rd) but is neutral on swing.

## Evals

"It looks right" isn't evidence, so the agent is measured on a fixed eval set: every question in `evals/cases.toml` runs against every demo (your own matches plus a public pro demo, so the numbers are reproducible).

```bash
uv run cs2-agent eval run --tag baseline        # runs every case, asks before spending
uv run cs2-agent eval report                    # metrics of the latest run
uv run cs2-agent eval label                     # label the judge's verdicts (blind)
uv run cs2-agent eval compare baseline latest   # what changed between two runs
uv run cs2-agent eval rejudge baseline          # same answers, re-judged by the current judge
uv run cs2-agent eval redteam                   # judge only, on conclusions with known right answers
```

What gets measured, and the question each metric answers:

| Metric | Question it answers |
| --- | --- |
| Facts verified / failed / unchecked | Of the facts shown to the user, how many are provably right? |
| First drafts with a wrong fact | How often the agent's first answer had a wrong fact, i.e. what you'd see with **no verifier**. The gap to "final answers with a wrong fact" is what verification buys. |
| Interpretations supported / rejected | How many conclusions survive the judge, and whether they fail on evidence (verifier) or on reasoning (judge) |
| Unbacked summary text | How often the answer slips in specifics that no claim backs |
| Summary contradicts a claim | How often the answer states something its own verified claims show is wrong |
| Before the critic | The same two numbers on the judge's first review, i.e. what you'd see with **no critique pass** |
| Judge agrees with expert | How often the judge's verdict matches a human player's, and whether it errs lenient or strict |
| Cost, latency, tool calls, cache hits | What a question costs to answer |

**Labelling.** `eval label` shows each interpretation with the verified facts behind it and asks whether the facts support it. The judge's own verdict stays hidden so it can't anchor you. Labels go to `evals/labels.jsonl` (committed, it's the human-labelled dataset) and are reused by later runs, so judge agreement is scored automatically from then on.

**Judge agreement needs a baseline.** Labels are lopsided (in the first set, 13 of 16 conclusions were fair), so a judge that rubber-stamps everything would score 81%. The report shows that trivial score next to the real one; a judge has to beat it.

**Red-teaming the judge.** Real answers are mostly fair (20 of 21 labelled "supported" in the latest run), so an approve-everything judge matches the labels too, and agreement alone can't show the judge rejects anything. `evals/redteam.toml` is an answer key built from real Falcons facts and reviewed by the expert: 46 conclusions, 26 of them deliberately bad (overreach, contradicting a fact, a role or intent the stats don't measure, a pattern from one match, an invented cause, judging a player by rating alone), plus mistakes planted in summaries and clean summaries to catch false alarms. It has two tiers. The easy tier is blatant mistakes; Haiku with judge v3 scored 100% on it, with one false alarm that turned out to be a real flaw in the answer key (the summary named the team, the fact didn't). The hard tier is subtle: the three real conclusions the judge got wrong against the expert labels, plus plausible-sounding overreach and causes the facts almost support.

Each prompt run 3 times on the same set (`--repeats 3`):

| Judge (Haiku 5.5) | Balanced accuracy (3 runs) | Spread | Easy | Hard | Planted mistakes caught | Clean summaries flagged |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 92.2% (93.1 / 93.1 / 90.6) | 2.5 pts | 98.8% | 77.1% | 24/24 | 0/24 |
| v4 | **99.4%** (100 / 100 / 98.1) | 1.9 pts | 100% | 97.9% | 24/24 | 0/24 |

Both judges caught every subtle trap and every planted summary mistake. v3 missed the same three items every time: the three real disagreements with the expert, so the red team reproduces the production failures. One of them was a prompt bug. Judge v2 added the rule "a match with 0-2 opening duels shows entrying wasn't their job; that count is the evidence itself, not a small sample". When the agent wrote that m0NESY's duels were "too few to show a pattern", the judge cited that rule as its reason to *reject* the conclusion the rule was written to accept. Judge v4 rewrites the rule around the conclusion instead of the count, and adds the expert's other two calls (converted opening wins are a fair basis; a "below average" verdict from rating alone is unfair when the facts show clutch wins or converted openings).

**A single run was misleading.** The first single run of v4 also scored 93.1%, with two fair conclusions newly rejected, which read as v4 only moving errors around. Run 3 more times, those two rejections never came back, and v4 beat v3 by 7 points against a run-to-run spread of about 2. So `--repeats N` now reports the spread between identical runs and how many verdicts flipped: a prompt change smaller than that spread hasn't been shown to do anything, and a single run can point either way.

**The caveat that remains:** v4's gain is entirely on the three items it was written from. On every other item v3 and v4 are both near-perfect, so this shows v4 fixed its targets without breaking anything else, not that it generalises. Fresh expert labels from the next full eval are the real test. `cs2-agent eval redteam` runs only the judge over it (about a cent with Haiku) and reports bad conclusions rejected, fair ones accepted, **balanced accuracy** (the mean of the two, so always-yes and always-no both score 50), planted mistakes caught, whether contradictions were filed as contradictions, and false alarms on clean summaries, then lists every miss.

**Comparing judges fairly.** `eval rejudge <run>` keeps a run's answers, resets only the judge's verdicts and asks the judge again, so two judge prompts rule on exactly the same labelled items. Judge prompts are versioned (`--judge-prompt v1|v2|v3|v4`); v2 was written after the first labels showed v1 rejecting fair reads 7 times out of 8 disagreements, v3 separates contradictions from unbacked text and stops flagging hedges, and v4 (the default) fixes a rule the red team showed backfiring. Because v2 was shaped by those labels, its score on them is optimistic; new labels from new demos are the real test.

**Runs** are written to `evals/runs/<run>/`: `results.jsonl` (one line per case, appended as each finishes), a trace per case, and `report.md`. A run that's interrupted or hits API errors can be finished with `--resume <run>`, which only runs the missing cases. One failing case never stops the run.

**Comparing runs with different question sets.** `eval compare` only uses the cases both runs answered, so adding questions doesn't skew a comparison (`--all` includes everything).

**Changing something?** Run `eval run --tag <change>` and `eval compare` against the baseline. Prompt tweaks, models (`--model claude-haiku-5-5`), step limits and the judge (`--no-judge`) are all flags.

### Results so far

| Run | What changed | Facts verified | Interpretations supported | Summary flagged | Cost / question |
| --- | --- | --- | --- | --- | --- |
| baseline | first eval (FACEIT + pro demo, 18 cases) | 100% | 41% | 67% | $0.028 |
| prompt-v2 | summary written from claims, proportionate conclusions, judge v2 | 100% | 87% | 22% | $0.033 |
| haiku | same, Haiku 5.5 as the agent | 100% | 63% | 50% | $0.003 |
| falcons | 3 Spirit vs Falcons maps, 18 cases | 100% | 87% | 56% | $0.035 |
| falcons-v3 | counting tools, critique pass, judge v3, 4 new questions (30 cases) | 100% | 100% | 3% (47% before the critique) | $0.054 |

The Falcons run is why the counting tools and the critique pass exist: on 30-round pro matches, the agent grouped and counted long lists in its head, and those numbers never became checkable claims. Every claim was right; the summaries weren't. In falcons-v3 the verifier caught its first wrong facts (3 drafts misread the new utility paths, all fixed before the final answer) and the judge caught 2 summaries contradicting their own claims. Two checks on that 3%, because the judge that writes the critique also grades the result:

- **What actually moved the number.** Re-judging the old Falcons answers with judge v3 drops them from 56% to 33% flagged, and the new answers' first drafts were flagged 39% of the time under the same judge. So the agent's first drafts weren't cleaner: the switch to judge v3 explains the first drop, and the critique pass explains the rest (39% to 3%). What did change in the drafts is the kind of mistake. Hand-counted totals are gone: every count now comes from a tool.
- **An independent read.** All 30 final answers were reviewed by hand against their own claims and against each other: 2 unbacked statements (both mild), 0 contradictions, and every cross-answer total consistent (untraded deaths are a subset of all deaths per killer, side splits add up, buy-type wins add up to the final score). The Haiku judge also flags statements that are backed word for word, so its flags are a signal, not ground truth.

## Tools

| Tool | What it answers |
| --- | --- |
| `list_demos` | Which demos are already parsed |
| `ingest_demo` | Parse a `.dem` file, returns the `demo_id` |
| `get_match_overview` | Score per team with CT/T split, scoreboard ranked by rating (HLTV 2.0 approximation) |
| `get_round_summary` | One round: buys, every player's loadout, kill feed with timings, bomb events |
| `get_opening_duels` | A player's first-kill duels by side, round conversion, trades, positions |
| `get_round_list` | The whole match round by round: winner, first kill, plant, each team's buy, running score; halves (overtime too), rounds won per buy type, first-kill conversion |
| `get_utility_stats` | Grenades thrown by type, side and before/after the plant, compared with teammates; flashes: enemy vs team vs self blinds (minor blinds under 1.5s separated), costly team flashes; nade damage |
| `get_multikills` | 2k / 3k / 4k / aces per player with round numbers, victims and weapons ("when did I get the ace?") |
| `get_clutches` | Clutches counted the way players do (won at any odds; lost only when attempted), with saves and hopeless last-alive rounds listed apart; totals per situation and per player |
| `get_win_probability` | One round as a win probability timeline: T's chance after every kill, plant and defuse, each event's swing, and the factors behind each estimate |
| `get_swing_ratings` | Round swing per player (percentage points per round) next to rating, with biggest plays and costliest moments |
| `find_kills` | Filterable, paginated kill feed with counts over all matches: by attacker, victim, side and round, plus opening/trade/round-won tags |
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

### Train the win probability model (free, no API)

```bash
uv run cs2-agent model train --ingest          # parse every demo in CS2_DEMO_DIR, train, print the CV report
uv run cs2-agent model swing match.dem         # swing per round vs rating for every player
uv run cs2-agent model swing match.dem --round 19   # one round, step by step
```

The model lives in the cache folder next to the parsed demos. Retrain whenever you add demos; the agent's `get_win_probability` and `get_swing_ratings` tools use it.

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

`uv run pytest` runs 182 tests with no network or API key:

- a hand-scripted 2v2 match that goes through the real ingest pipeline
- mocked HTTP for the FACEIT and Leetify clients
- an in-memory MCP session for the server
- the agent loop driven by a scripted fake model: revisions, the critique pass, the plain-text nudge, the step limit, refused tool calls after the final step, cost accounting and trace files
- the eval runner, metrics, resume, labelling and reports, with a scripted model that gets a fact wrong and fixes it after feedback

For a check against a real match with a known result:

```bash
CS2_TEST_DEMO=/path/to/faceit-de_mirage-6de46e85.dem uv run pytest -m demo
```

That demo is public in [awpy-fixtures](https://github.com/pnxenopoulos/awpy-fixtures). The tests check the 16-14 score, FACEIT's published kills and deaths, that the opening-duel, clutch and grenade logic agree with awpy's independent scoreboard for all 10 players, the overtime halves, and that agent evidence paths resolve against real tool output.

## Roadmap

- [x] Phase 1: tool layer
- [x] Phase 2: agent loop with findings schema, deterministic verifier, judge and tracing
- [x] Phase 3: eval runner, metrics with a no-verifier baseline, blind labelling and judge agreement
- [ ] Phase 3b: labelled "key mistakes" per match, to measure whether the agent finds what matters
- [x] Phase 4: explainable round win probability model (XGBoost + logistic ensemble, TreeSHAP) and round swing rating
- [ ] Phase 5: web app

Built on [awpy](https://github.com/pnxenopoulos/awpy), the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and the [Anthropic API](https://platform.claude.com/docs).
