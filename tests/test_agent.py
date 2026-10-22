"""The agent loop end to end, with a scripted fake model and the real MCP server on the fake demo.

No API key or network: FakeLLM replays a fixed list of responses and records
what the loop sent it, so we can assert on both directions.
"""

import copy
import json

import anyio
import pytest

from cs2_analyst.agent import AgentConfig, run_agent
from cs2_analyst.agent import cli
from cs2_analyst.agent.llm import Block, LLMResponse
from cs2_analyst.agent.prompts import FINAL_STEP, NO_REPORT
from cs2_analyst.agent.toolbox import Toolbox
from cs2_analyst.config import Settings
from cs2_analyst.server import build_server

from .conftest import DEMO_ID

USAGE = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
DEMO = {"demo_id": DEMO_ID, "map": "de_test", "rounds_played": 4}


def tool(name, **args):
    return {"type": "tool_use", "id": f"tu_{name}_{json.dumps(args, sort_keys=True)}", "name": name, "input": args}


def text(s):
    return {"type": "text", "text": s}


def report(*claims, summary="answer"):
    return tool("submit_report", summary=summary, claims=list(claims))


def fact(cid, text_, *checks):
    return {"id": cid, "text": text_, "kind": "fact", "supports": [],
            "checks": [{"call_id": c, "path": p, "expected": e} for c, p, e in checks]}


def interp(cid, text_, *supports):
    return {"id": cid, "text": text_, "kind": "interpretation", "checks": [], "supports": list(supports)}


class FakeLLM:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.calls = []

    async def create(self, *, model, system, tools, messages, max_tokens):
        self.calls.append({"model": model, "tools": [t["name"] for t in tools], "messages": copy.deepcopy(messages)})
        blocks = self.turns.pop(0)
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return LLMResponse([Block(**b) for b in blocks], blocks, dict(USAGE), stop)


def last_user_text(call) -> str:
    content = call["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(c.get("content") or c.get("text") or "" for c in content)


@pytest.fixture
def run(store):
    settings = Settings(cache_dir=store.cache_dir, demo_dir=None, faceit_api_key=None,
                        leetify_api_key=None, http_cache_ttl_s=0)

    def _run(llm, **cfg):
        async def go():
            async with Toolbox.connect(build_server(settings=settings, store=store)) as tb:
                return await run_agent("Why did Team A lose?", DEMO, llm, tb, AgentConfig(**cfg))
        return anyio.run(go)

    return _run


def test_full_run_with_one_revision_and_judge(run):
    wrong_then_right = lambda expected: fact("c2", "alpha won his opening duels", ("t2", "overall.won", expected))
    llm = FakeLLM(
        [text("Checking the overview first."), tool("get_match_overview", demo_id=DEMO_ID)],
        [tool("get_opening_duels", demo_id=DEMO_ID, player="alpha")],
        [report(
            fact("c1", "Team B won 3 rounds", ("t1", "score.Team B (started T).rounds_won", "3")),
            wrong_then_right("3"),
            fact("c3", "alpha's round 4 opening death was traded", ("t2", "duels[round=4].traded", "true")),
            interp("c4", "Team A kept losing the opening fight", "c1", "c3"),
            fact("c5", "a fact with no evidence"),
        )],
        [report(
            fact("c1", "Team B won 3 rounds", ("t1", "score.Team B (started T).rounds_won", "3")),
            wrong_then_right("2"),
            fact("c3", "alpha's round 4 opening death was traded", ("t2", "duels[round=4].traded", "true")),
            interp("c4", "Team A kept losing the opening fight", "c1", "c3"),
            fact("c5", "a fact with no evidence"),
        )],
        # judge turn
        [tool("submit_verdicts", verdicts=[{"claim_id": "c4", "supported": True, "reason": "fits"}])],
    )
    res = run(llm)

    assert res.stop_reason == "submitted" and res.revisions == 1
    assert {r.claim.id: r.status for r in res.verification.results} == {
        "c1": "verified", "c2": "verified", "c3": "verified", "c4": "supported", "c5": "unchecked"}

    # What the loop sent the model
    first = llm.calls[0]
    assert "submit_report" in first["tools"] and "ingest_demo" not in first["tools"]
    assert DEMO_ID in first["messages"][0]["content"]
    assert "[call_id=t1]" in last_user_text(llm.calls[1])
    assert "actual value: 2" in last_user_text(llm.calls[3]), "failed claim must come back with the real value"
    assert llm.calls[4]["model"] == "claude-haiku-5-5", "judge uses the cheaper model"

    # Trace and cost: 4 Sonnet turns at (1000*$2 + 200*$10)/1M, plus 1 Haiku judge turn
    s = res.trace.summary()
    assert (s["llm_calls"], s["tool_calls"]) == (5, 2)
    assert s["cost_usd"] == pytest.approx(4 * 0.004 + (1000 * 0.10 + 200 * 0.50) / 1e6)
    assert list(res.ledger) == ["t1", "t2"]


def test_plain_text_answer_is_nudged_then_accepted(run):
    llm = FakeLLM(
        [text("They lost because of bad opening duels.")],
        [report(summary="ok")],
    )
    res = run(llm, judge_model=None)
    assert NO_REPORT in last_user_text(llm.calls[1])
    assert res.stop_reason == "submitted"


def test_step_limit_forces_a_final_turn_and_marks_text_unverified(run):
    llm = FakeLLM(
        [tool("get_match_overview", demo_id=DEMO_ID)],
        [tool("get_round_summary", demo_id=DEMO_ID, round=2)],
        [text("Out of steps, here's my best guess.")],
    )
    res = run(llm, max_steps=2, judge_model=None)
    assert FINAL_STEP in last_user_text(llm.calls[2])
    assert res.stop_reason == "text_answer_unverified"
    assert res.report.summary.startswith("Out of steps") and res.report.claims == []


def test_tools_are_refused_after_the_final_step(run):
    llm = FakeLLM(
        [tool("get_match_overview", demo_id=DEMO_ID)],
        [tool("get_round_summary", demo_id=DEMO_ID, round=1)],  # ignores the final-step instruction
    )
    res = run(llm, max_steps=1, judge_model=None)
    assert list(res.ledger) == ["t1"], "no tool may run after the final step"
    assert res.report is None and res.stop_reason == "step_limit"


def test_failed_tool_calls_are_recorded_and_cannot_be_cited(run):
    llm = FakeLLM(
        [tool("get_opening_duels", demo_id=DEMO_ID, player="zulu")],
        [report(fact("c1", "zulu won 3 duels", ("t1", "overall.won", "3")))],
    )
    res = run(llm, judge_model=None, max_revisions=0)
    assert res.ledger["t1"]["is_error"] and "Players:" in res.ledger["t1"]["result"]
    assert res.verification.results[0].status == "failed"


def test_trace_file_is_jsonl_with_summary(run, tmp_path):
    res = run(FakeLLM([report(summary="ok")]), judge_model=None)
    lines = [json.loads(l) for l in res.trace.write(tmp_path).read_text().splitlines()]
    assert lines[0]["event"] == "run_start" and lines[-1]["event"] == "summary"
    assert any(e["event"] == "llm_call" and e["usage"]["input_tokens"] == 1000 for e in lines)


def test_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main(["ask", "x.dem", "q"])


def test_render_keeps_brackets_in_paths_and_names(run):
    """Rich treats [..] as markup; data like filter paths must be escaped, not swallowed."""
    from rich.console import Console

    llm = FakeLLM(
        [tool("get_match_overview", demo_id=DEMO_ID)],
        [report(fact("c1", "[clan] alpha got 9 kills", ("t1", "scoreboard[name=alpha].kills", "9")))],
    )
    res = run(llm, judge_model=None, max_revisions=0)
    console = Console(record=True, width=200, color_system=None)
    cli.render(res, console)
    out = console.export_text()
    assert "scoreboard[name=alpha].kills" in out and "[clan] alpha" in out


def test_cli_missing_demo_is_one_clean_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("CS2_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CS2_DEMO_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        cli.main(["ask", "nope.dem", "q"])
    assert str(e.value).startswith("Could not load demo: No file 'nope.dem' in CS2_DEMO_DIR")


def test_cli_explains_unset_demo_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("CS2_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CS2_DEMO_DIR", "")  # e.g. a blank duplicate line in .env
    with pytest.raises(SystemExit, match="CS2_DEMO_DIR is not set"):
        cli.main(["ask", "nope.dem", "q"])


def test_cli_api_errors_are_one_clean_line(monkeypatch, store):
    import httpx
    from anthropic import AuthenticationError

    class BadKeyLLM:
        async def create(self, **kw):
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=req), body=None)

    from cs2_analyst.demo_store import ingest_demo_object

    from .conftest import FakeDemo

    hex_id = "abcdef012345"  # real ids are 12 hex chars; that's how the CLI tells ids from file names
    ingest_demo_object(FakeDemo(), hex_id, store.cache_dir / hex_id, source="fake.dem")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("CS2_CACHE_DIR", str(store.cache_dir))
    monkeypatch.setattr(cli, "AnthropicLLM", BadKeyLLM)
    with pytest.raises(SystemExit, match="Anthropic API error 401: invalid x-api-key"):
        cli.main(["ask", hex_id, "q"])


def test_judge_flags_unbacked_summary_statements(run):
    """The summary can't sneak in specifics (a ranking, a name) that no verified claim backs."""
    from rich.console import Console

    llm = FakeLLM(
        [tool("get_match_overview", demo_id=DEMO_ID)],
        [report(fact("c1", "alpha got 2 kills", ("t1", "scoreboard[name=alpha].kills", "2")),
                summary="alpha got 2 kills and was the best player on his team.")],
        [tool("submit_verdicts", verdicts=[], unbacked_summary_statements=["alpha was the best player on his team"])],
    )
    res = run(llm)
    assert res.verification.summary_issues == ["alpha was the best player on his team"]
    judge_input = json.loads(llm.calls[2]["messages"][0]["content"])
    assert judge_input["summary"].startswith("alpha got 2 kills")
    assert judge_input["verified_claims"] == ["alpha got 2 kills"]

    console = Console(record=True, width=200, color_system=None)
    cli.render(res, console)
    assert "not backed by a verified claim" in console.export_text()
