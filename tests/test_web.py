"""Phase 5 web API on the fake 2v2 match. No network: the radar transform and replay cache are faked."""

from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from cs2_analyst.config import Settings
from cs2_analyst.web import cs2
from cs2_analyst.web import replay as R
from cs2_analyst.web.app import create_app

from .conftest import DEMO_ID

TRANSFORM = {"pos_x": -1000.0, "pos_y": 1000.0, "scale": 2.0}


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    demo_dir = tmp_path / "demos"
    demo_dir.mkdir()
    (demo_dir / "new.dem").write_bytes(b"x" * 10)
    settings = Settings(cache_dir=store.cache_dir, demo_dir=demo_dir, faceit_api_key=None, leetify_api_key=None,
                        http_cache_ttl_s=0)
    monkeypatch.setattr(R, "map_transform", lambda name: TRANSFORM)
    return TestClient(create_app(settings=settings, store=store))


def fake_replay_cache(store):
    d = store.cache_dir / DEMO_ID
    rows = [{"tick": t, "steamid": sid, "name": n, "side": side, "x": 100.0 + t, "y": 200.0, "z": 0.0, "yaw": 90.0,
             "health": 100 if t < 400 or sid != 3 else 0, "active_weapon": "ak47", "has_bomb": sid == 3,
             "is_defusing": False, "flash_duration": 0.0}
            for t in (200, 300, 400, 500)
            for sid, n, side in [(1, "alpha", "CT"), (2, "bravo", "CT"), (3, "charlie", "T"), (4, "delta", "T")]]
    pl.DataFrame(rows, schema_overrides={"steamid": pl.UInt64}).write_parquet(d / "replay_positions.parquet")
    pl.DataFrame({"start_tick": [250], "end_tick": [900], "x": [0.0], "y": [0.0]}).write_parquet(d / "replay_smokes.parquet")
    pl.DataFrame({"start_tick": [5000], "end_tick": [5100], "x": [0.0], "y": [0.0]}).write_parquet(d / "replay_fires.parquet")


def test_demo_list_and_folder_files(client):
    demos = client.get("/api/demos").json()
    assert [d["demo_id"] for d in demos] == [DEMO_ID] and demos[0]["source_name"] == "fake.dem"
    files = client.get("/api/files").json()
    assert files["files"] == [{"name": "new.dem", "size_mb": 0.0, "parsed": False}]


def test_overview_rounds_and_errors(client):
    ov = client.get(f"/api/demos/{DEMO_ID}/overview").json()
    assert ov["score"]["Team B (started T)"]["rounds_won"] == 3 and "multikills" in ov
    assert len(client.get(f"/api/demos/{DEMO_ID}/rounds").json()["rounds"]) == 4
    bad = client.get("/api/demos/nope/overview")
    assert bad.status_code == 400 and "Unknown demo_id" in bad.json()["detail"]


def test_model_endpoints_explain_how_to_train(client):
    assert client.get("/api/model").json()["trained"] is False
    r = client.get(f"/api/demos/{DEMO_ID}/swing")
    assert r.status_code == 400 and "cs2-agent model train" in r.json()["detail"]


def test_replay_needs_the_original_demo(client):
    r = client.get(f"/api/demos/{DEMO_ID}/rounds/1/replay")
    assert r.status_code == 400 and "original demo file" in r.json()["detail"]


def test_replay_frames_in_radar_pixels(client, store):
    fake_replay_cache(store)
    rp = client.get(f"/api/demos/{DEMO_ID}/rounds/1/replay").json()
    assert rp["ticks"] == [200, 300, 400, 500] and rp["units_per_px"] == 2.0
    alpha = next(p for p in rp["players"] if p["name"] == "alpha")
    # x=300 -> (300 - -1000) / 2 = 650 ; y=200 -> (1000 - 200) / 2 = 400
    assert alpha["frames"][0][:4] == [650.0, 400.0, 90, 100]
    charlie = next(p for p in rp["players"] if p["name"] == "charlie")
    assert charlie["frames"][2] is None, "dead players aren't drawn"
    assert [k["victim"] for k in rp["kills"]] == ["charlie", "alpha", "delta", "charlie"]
    assert len(rp["smokes"]) == 1 and rp["fires"] == [], "only grenades active during the round"
    assert rp["radar_url"] == "/api/maps/de_test/radar.png"
    assert rp["players"] and all(p["name"] != "coach" for p in rp["players"])


def test_cs2_commands_jump_five_seconds_early(client, tmp_path, monkeypatch):
    monkeypatch.setattr(cs2, "csgo_dir", lambda: None)
    info = client.get(f"/api/demos/{DEMO_ID}/cs2?tick=1000").json()
    assert info == {"game_dir": None, "prepared": False, "commands": ["playdemo cs2-analyst/fake", "demo_gototick 680"]}
    r = client.post(f"/api/demos/{DEMO_ID}/cs2/prepare")
    assert r.status_code == 400 and "CS2_GAME_DIR" in r.json()["detail"]


def test_cs2_prepare_copies_once_with_a_console_safe_name(tmp_path):
    src = tmp_path / "spirit vs falcons (m1).dem"
    src.write_bytes(b"demo")
    game = tmp_path / "csgo"
    game.mkdir()
    target = cs2.prepare(src, game)
    assert target == game / "cs2-analyst" / "spirit_vs_falcons__m1_.dem" and target.read_bytes() == b"demo"
    assert cs2.commands(src, None, 64.0) == ["playdemo cs2-analyst/spirit_vs_falcons__m1_"]
    mtime = target.stat().st_mtime_ns
    cs2.prepare(src, game)
    assert target.stat().st_mtime_ns == mtime, "an identical copy isn't copied again"


def test_csgo_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS2_GAME_DIR", str(tmp_path))
    assert cs2.csgo_dir() == tmp_path


def test_upload_rejects_non_demos(client):
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hi")})
    assert r.status_code == 400 and ".dem" in r.json()["detail"]
