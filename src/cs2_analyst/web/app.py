"""FastAPI backend for the web app. Thin, like server.py: every route calls the same analysis
functions, model and cache the MCP tools use, so the app and the agent can never disagree.

Run it with `cs2-agent web`. It serves the built frontend (web/dist) at / and the JSON API under /api.
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import analysis
from ..config import Settings, get_settings
from ..demo_store import DemoStore
from ..errors import AnalystError
from . import cs2
from . import replay as R

DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class IngestBody(BaseModel):
    file: str


def create_app(settings: Settings | None = None, store: DemoStore | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = store or DemoStore(settings.cache_dir)
    app = FastAPI(title="cs2-analyst", version="0.1.0")
    # Old caches (e.g. with coaches counted as players) are re-parsed once, before anything reads them.
    store.refresh_stale(lambda n: print(f"re-parsing {n} (newer cache format)..."))

    @app.exception_handler(AnalystError)
    async def analyst_error(_: Request, e: AnalystError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    def table(demo_id: str, name: str):
        return store.table(demo_id, name)

    @functools.cache
    def winprob_model():
        from ..winprob import model as wp

        return wp.load(settings.cache_dir / "winprob")

    def swing_states(demo_id: str):
        from ..winprob import swing as sw
        from ..winprob.features import states_for

        return sw.with_probabilities(winprob_model()[0], states_for(store, demo_id))

    # ------------------------------------------------------------ demos

    @app.get("/api/demos")
    def demos() -> list[dict[str, Any]]:
        out = []
        for d in store.list_demos():
            meta = store.meta(d["demo_id"])
            out.append({**d, "source_name": Path(d.get("source") or "").name, "max_rounds": meta.get("max_rounds")})
        return sorted(out, key=lambda d: d.get("ingested_at") or "", reverse=True)

    @app.get("/api/files")
    def files() -> dict[str, Any]:
        """.dem files in CS2_DEMO_DIR, and whether each is already parsed."""
        if not settings.demo_dir or not settings.demo_dir.is_dir():
            return {"demo_dir": str(settings.demo_dir) if settings.demo_dir else None, "files": []}
        parsed = {Path(d.get("source") or "").name for d in store.list_demos()}
        found = [{"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1), "parsed": f.name in parsed}
                 for f in sorted(settings.demo_dir.glob("*.dem"))]
        return {"demo_dir": str(settings.demo_dir), "files": found}

    @app.post("/api/ingest")
    def ingest(body: IngestBody) -> dict[str, Any]:
        p = Path(body.file).expanduser()
        if not p.is_absolute() and settings.demo_dir:
            p = settings.demo_dir / p
        return store.ingest(p)

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        if not (file.filename or "").lower().endswith(".dem"):
            raise HTTPException(400, "Upload a .dem file (extract .dem.gz / .dem.zst downloads first).")
        folder = settings.demo_dir or (settings.cache_dir / "uploads")
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / Path(file.filename).name
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        return store.ingest(dest)

    # ------------------------------------------------------------ match

    @app.get("/api/demos/{demo_id}/overview")
    def overview(demo_id: str) -> dict[str, Any]:
        ov = analysis.match_overview(
            store.meta(demo_id), table(demo_id, "rounds"), table(demo_id, "stats"), table(demo_id, "players"),
            table(demo_id, "round_players"), kills=table(demo_id, "kills"), damages=table(demo_id, "damages"),
            bomb=table(demo_id, "bomb"))
        return {**ov, "multikills": analysis.multikills(table(demo_id, "kills"), table(demo_id, "rounds"), min_kills=3)}

    @app.get("/api/demos/{demo_id}/rounds")
    def rounds(demo_id: str) -> dict[str, Any]:
        return analysis.round_list(
            table(demo_id, "rounds"), table(demo_id, "kills"), table(demo_id, "round_players"),
            table(demo_id, "players"), table(demo_id, "round_economy"), table(demo_id, "bomb"),
            max_rounds=store.meta(demo_id).get("max_rounds"))

    @app.get("/api/model")
    def model_info() -> dict[str, Any]:
        from ..winprob import model as wp

        try:
            _, meta = winprob_model()
        except wp.ModelMissing as e:
            return {"trained": False, "message": str(e)}
        return {"trained": True, **{k: meta.get(k) for k in ("demos", "rounds", "states", "scores", "calibration",
                                                           "trained_at")}}

    @app.get("/api/demos/{demo_id}/swing")
    def swing(demo_id: str) -> dict[str, Any]:
        from ..winprob import swing as sw

        rp = table(demo_id, "round_players")
        labels = analysis.team_labels(rp, table(demo_id, "players"))
        ratings = dict(analysis.with_rating(table(demo_id, "stats")).select("steamid", "rating").iter_rows())
        return sw.swing_ratings(swing_states(demo_id), rp, labels, ratings)

    @app.get("/api/demos/{demo_id}/rounds/{round_num}/winprob")
    def winprob(demo_id: str, round_num: int) -> dict[str, Any]:
        from ..winprob import swing as sw

        tl = sw.round_timeline(swing_states(demo_id), winprob_model()[0], round_num, store.meta(demo_id)["tick_rate"])
        if not tl:
            raise AnalystError(f"No win probability for round {round_num}.")
        return tl

    @app.get("/api/demos/{demo_id}/rounds/{round_num}/replay")
    def round_replay(demo_id: str, round_num: int) -> dict[str, Any]:
        meta = store.meta(demo_id)
        data = R.round_replay(store, demo_id, round_num, R.map_transform(meta["map"]))
        return {**data, "radar_url": f"/api/maps/{meta['map']}/radar.png"}

    @app.get("/api/maps/{map_name}/radar.png")
    def radar(map_name: str) -> FileResponse:
        return FileResponse(R.radar_file(map_name), media_type="image/png")

    # ------------------------------------------------------------ watch in CS2

    @app.get("/api/demos/{demo_id}/cs2")
    def cs2_info(demo_id: str, tick: int | None = None) -> dict[str, Any]:
        meta = store.meta(demo_id)
        source = Path(meta.get("source") or "")
        game = cs2.csgo_dir()
        return {
            "game_dir": str(game) if game else None,
            "prepared": bool(game and cs2.prepared_path(source, game).exists()),
            "commands": cs2.commands(source, tick, meta["tick_rate"]),
        }

    @app.post("/api/demos/{demo_id}/cs2/prepare")
    def cs2_prepare(demo_id: str) -> dict[str, Any]:
        source = Path(store.meta(demo_id).get("source") or "")
        target = cs2.prepare(source, cs2.csgo_dir())
        return {"prepared": True, "path": str(target)}

    # ------------------------------------------------------------ frontend

    if DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            f = DIST / path
            return FileResponse(f if path and f.is_file() else DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend() -> dict[str, str]:
            return {"detail": "Frontend not built. Run: cd web && npm install && npm run build"}

    return app
