"""Phase 4: game states, the win probability model and swing credit. No network, no API key."""

import json

import numpy as np
import polars as pl
import pytest

from cs2_analyst.winprob import model as M
from cs2_analyst.winprob import swing as S
from cs2_analyst.winprob.features import FEATURES, MONOTONE, round_states, training_rows

from .conftest import DEMO_ID


@pytest.fixture
def states(tables):
    return round_states(DEMO_ID, tables("rounds"), tables("kills"), tables("damages"), tables("bomb"),
                        tables("round_players"), 64.0)


def test_one_state_per_event_starting_at_freeze_end(states):
    r1 = states.filter(pl.col("round") == 1).sort("step")
    # start, alpha kills charlie, delta kills alpha, bravo kills delta (the exit frag at 1100 is ignored)
    assert r1["event"].to_list() == ["start", "kill", "kill", "kill"]
    assert list(zip(r1["t_alive"], r1["ct_alive"])) == [(2, 2), (1, 2), (1, 1), (0, 1)]
    assert r1["t_win"].unique().to_list() == [0], "CT won round 1"
    assert r1["actor"].to_list()[1:] == ["alpha", "delta", "bravo"]
    assert r1["round_time_left_s"][0] == 115.0 and r1["round_time_left_s"][1] == 115.0 - (400 - 200) / 64


def test_plant_switches_the_clock_to_the_bomb_timer(states):
    r2 = states.filter(pl.col("round") == 2).sort("step")
    plant = r2.filter(pl.col("event") == "plant").row(0, named=True)
    assert plant["bomb_planted"] == 1 and plant["round_time_left_s"] == 0.0 and plant["bomb_time_left_s"] == 40.0
    assert plant["actor"] == "charlie" and plant["actor_side"] == "T"
    assert r2["t_win"][0] == 1


def test_health_and_equipment_of_players_still_alive(tables):
    dmg = pl.DataFrame({"tick": [350], "round_num": [1], "victim_steamid": [3], "health_post": [60],
                        "after_round_end": [False]}, schema_overrides={"victim_steamid": pl.UInt64, "round_num": pl.Int32})
    s = round_states(DEMO_ID, tables("rounds"), tables("kills"), dmg, tables("bomb"), tables("round_players"), 64.0)
    r1 = s.filter(pl.col("round") == 1).sort("step")
    assert r1["t_hp"][0] == 200, "full health at freeze end"
    # charlie was hit to 60 at tick 350, then killed at 400: only delta's 100 is left
    assert r1["t_hp"][1] == 100 and r1["t_equip"][1] == 1000


def test_training_rows_drop_decided_states(states):
    tr = training_rows(states)
    assert ((tr["t_alive"] > 0) & (tr["ct_alive"] > 0)).all()
    assert "defuse" not in tr["event"].to_list()


# ---------------------------------------------------------------- model

def synthetic_states(n_demos: int = 3, rounds: int = 60, seed: int = 0) -> pl.DataFrame:
    """Rounds where the side with more players alive and more equipment usually wins."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_demos):
        for r in range(rounds):
            t_alive, ct_alive = rng.integers(1, 6), rng.integers(1, 6)
            t_eq, ct_eq = rng.integers(1, 6) * 4000, rng.integers(1, 6) * 4000
            planted = int(rng.random() < 0.3)
            logit = 0.9 * (t_alive - ct_alive) + 0.0002 * (t_eq - ct_eq) + 1.2 * planted
            rows.append({
                "demo_id": f"d{d}", "round": r, "step": 1, "tick": 0, "event": "kill",
                "t_alive": t_alive, "ct_alive": ct_alive, "t_hp": 90 * t_alive, "ct_hp": 90 * ct_alive,
                "t_equip": t_eq, "ct_equip": ct_eq, "ct_defusers": int(rng.integers(0, ct_alive + 1)),
                "bomb_planted": planted, "round_time_left_s": 0.0 if planted else float(rng.uniform(10, 115)),
                "bomb_time_left_s": float(rng.uniform(5, 40)) if planted else 0.0,
                "t_win": int(rng.random() < 1 / (1 + np.exp(-logit))),
            })
    return pl.DataFrame(rows)


def test_cross_validation_beats_the_constant_and_reports_calibration():
    cv = M.cross_validate(synthetic_states())
    assert set(cv["scores"]) == {"constant", "players_alive", "logistic", "xgboost", "model"}
    assert cv["demos"] == 3 and cv["rounds"] == 180
    assert cv["scores"]["model"]["log_loss"] < cv["scores"]["constant"]["log_loss"]
    assert cv["scores"]["constant"]["auc"] is None, "AUC of a constant is meaningless"
    assert sum(b["states"] for b in cv["calibration"]) == cv["states"]


def test_cross_validation_needs_two_demos():
    with pytest.raises(ValueError, match="at least 2"):
        M.cross_validate(synthetic_states(n_demos=1))


def test_model_respects_monotone_constraints(tmp_path):
    M.train(synthetic_states(), tmp_path)
    model, meta = M.load(tmp_path)
    assert meta["features"] == FEATURES and set(meta["importance_mean_abs_contribution"]) == set(FEATURES)
    base = synthetic_states().head(1)
    for f, sign in MONOTONE.items():
        lo = M.predict(model, base.with_columns(pl.lit(1).alias(f)))[0]
        hi = M.predict(model, base.with_columns(pl.lit(4 if "alive" in f or f in ("ct_defusers",) else 30000
                                                      if "equip" in f else 400 if "hp" in f else 39)
                                               .alias(f)))[0]
        if sign > 0:
            assert hi >= lo - 1e-9, f"more {f} must not lower T's chance"
        elif sign < 0:
            assert hi <= lo + 1e-9, f"more {f} must not raise T's chance"


def test_predict_pins_decided_states(tmp_path):
    M.train(synthetic_states(), tmp_path)
    model, _ = M.load(tmp_path)
    s = synthetic_states().head(3).with_columns(
        pl.Series("ct_alive", [0, 2, 2]), pl.Series("t_alive", [2, 0, 2]), pl.Series("bomb_planted", [0, 0, 1]),
        pl.Series("event", ["kill", "kill", "defuse"]))
    assert M.predict(model, s).tolist() == [1.0, 0.0, 0.0]


def test_missing_model_says_how_to_train(tmp_path):
    with pytest.raises(M.ModelMissing, match="cs2-agent model train"):
        M.load(tmp_path)


def test_explanations_add_up_to_the_prediction(tmp_path):
    M.train(synthetic_states(), tmp_path)
    model, _ = M.load(tmp_path)
    X = synthetic_states().head(5).select(FEATURES).to_numpy().astype(float)
    from xgboost import DMatrix

    full = model.xgb.get_booster().predict(DMatrix(X, feature_names=FEATURES), pred_contribs=True)
    p = model.xgb.predict_proba(X)[:, 1]
    assert np.allclose(1 / (1 + np.exp(-full.sum(axis=1))), p, atol=1e-5), "TreeSHAP is exact: contributions sum to the logit"
    lin = model.linear.contributions(X).sum(axis=1) + model.linear.intercept
    assert np.allclose(1 / (1 + np.exp(-lin)), model.linear.predict_proba(X)), "and so are the linear ones"
    assert np.allclose(model.predict_proba(X), (p + model.linear.predict_proba(X)) / 2), "shipped = average"
    reasons = M.top_reasons(M.explain(model, X)[0], X[0])
    assert len(reasons) == 3 and all(r["pushes"] in ("T", "CT") for r in reasons)
    for r in reasons:
        assert r["reason"].endswith(f"-> {r['pushes']}") and r["level"] in ("high", "low")


def test_reasons_say_whether_a_value_is_high_or_low():
    from cs2_analyst.winprob.features import FEATURES as F

    contribs = np.zeros(len(F))
    contribs[F.index("t_equip")] = -0.8      # pushes CT: so T's equipment is low
    contribs[F.index("ct_hp")] = 0.5         # pushes T: so CT health is low (more CT health only helps CT)
    contribs[F.index("bomb_planted")] = 0.3
    row = np.zeros(len(F))
    row[F.index("bomb_planted")] = 1
    r = [x["reason"] for x in M.top_reasons(contribs, row)]
    assert r == ["low T equipment -> CT", "low CT health -> T", "bomb planted -> T"]


# ---------------------------------------------------------------- swing credit

def _timeline(p_t, events):
    rows = []
    for step, (p, ev) in enumerate(zip(p_t, events)):
        rows.append({"demo_id": "x", "round": 1, "step": step, "tick": step * 100, "p_t": p,
                     "event": ev.get("event", "start"), "actor": ev.get("actor"), "actor_id": ev.get("actor_id"),
                     "actor_side": ev.get("actor_side"), "victim": ev.get("victim"), "victim_id": ev.get("victim_id"),
                     "victim_side": ev.get("victim_side"), "assister": ev.get("assister"),
                     "assister_id": ev.get("assister_id")})
    df = pl.DataFrame(rows)
    return df.with_columns((pl.col("p_t") - pl.col("p_t").shift(1)).fill_null(0.0).alias("delta_t"))


KILL_T = {"event": "kill", "actor": "t1", "actor_id": 1, "actor_side": "T", "victim": "c1", "victim_id": 2, "victim_side": "CT"}


def test_killer_gains_what_the_victim_loses():
    c = S.credits(_timeline([0.5, 0.7], [{}, KILL_T]))
    by = {x["name"]: x["swing"] for x in c}
    assert by == pytest.approx({"t1": 0.2, "c1": -0.2})


def test_assister_takes_a_quarter_of_the_gain():
    ev = {**KILL_T, "assister": "t2", "assister_id": 3}
    by = {x["name"]: x["swing"] for x in S.credits(_timeline([0.5, 0.7], [{}, ev]))}
    assert by == pytest.approx({"t1": 0.15, "t2": 0.05, "c1": -0.2})


def test_plant_and_defuse_credit_the_player_who_did_it():
    events = [{}, {"event": "plant", "actor": "t1", "actor_id": 1, "actor_side": "T"},
              {"event": "defuse", "actor": "c1", "actor_id": 2, "actor_side": "CT"}]
    by = {x["role"]: x["swing"] for x in S.credits(_timeline([0.4, 0.75, 0.0], events))}
    assert by == pytest.approx({"plant": 0.35, "defuse": 0.75})


def test_teamkill_only_charges_the_victim():
    tk = {**KILL_T, "victim": "t2", "victim_id": 3, "victim_side": "T"}
    c = S.credits(_timeline([0.5, 0.35], [{}, tk]))
    assert [(x["name"], x["role"], round(x["swing"], 2)) for x in c] == [("t2", "death", -0.15)]


def test_swing_ratings_on_the_fake_match(tables, tmp_path):
    from cs2_analyst import analysis as A

    M.train(synthetic_states(), tmp_path)
    model, _ = M.load(tmp_path)
    st = S.with_probabilities(model, round_states(DEMO_ID, tables("rounds"), tables("kills"), tables("damages"),
                                                  tables("bomb"), tables("round_players"), 64.0))
    labels = A.team_labels(tables("round_players"), tables("players"))
    sw = S.swing_ratings(st, tables("round_players"), labels, {1: 0.9, 2: 1.3, 3: 0.7, 4: 1.1})
    names = [p["name"] for p in sw["players"]]
    assert sorted(names) == ["alpha", "bravo", "charlie", "delta"]
    assert [p["swing_rank"] for p in sw["players"]] == [1, 2, 3, 4]
    assert all(p["rounds"] == 4 for p in sw["players"])
    # credits are zero-sum per kill, so the four players' totals add up to what the plants/defuses added
    bomb = sum(p["from_bomb_pp"] for p in sw["players"])
    assert sum(p["total_swing_pp"] for p in sw["players"]) == pytest.approx(bomb, abs=0.5)
    tl = S.round_timeline(st, model, 1, 64.0)
    assert tl["winner_side"] == "CT" and len(tl["steps"]) == 4 and tl["steps"][-1]["p_t_win"] == 0.0
    json.dumps(tl), json.dumps(sw)  # tool results must be JSON-serialisable


def test_tools_through_protocol(store, tmp_path):
    import anyio
    from mcp import Client

    from cs2_analyst.config import Settings
    from cs2_analyst.server import build_server

    settings = Settings(cache_dir=store.cache_dir, demo_dir=None, faceit_api_key=None, leetify_api_key=None,
                        http_cache_ttl_s=0)

    async def call(name, args):
        async with Client(build_server(settings=settings, store=store, log_level="WARNING")) as c:
            return await c.call_tool(name, args)

    missing = anyio.run(call, "get_swing_ratings", {"demo_id": DEMO_ID})
    assert missing.is_error and "cs2-agent model train" in missing.content[0].text

    M.train(synthetic_states(), store.cache_dir / "winprob")
    sw = json.loads(anyio.run(call, "get_swing_ratings", {"demo_id": DEMO_ID}).content[0].text)
    assert len(sw["players"]) == 4
    wp = anyio.run(call, "get_win_probability", {"demo_id": DEMO_ID, "round": 2})
    assert json.loads(wp.content[0].text)["winner_side"] == "T"
    bad = anyio.run(call, "get_win_probability", {"demo_id": DEMO_ID, "round": 9})
    assert bad.is_error and "Valid rounds" in bad.content[0].text
