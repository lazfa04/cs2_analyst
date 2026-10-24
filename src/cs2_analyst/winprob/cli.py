"""`cs2-agent model ...`: train the win probability model and look at swing ratings. No API calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..config import get_settings
from ..demo_store import DemoStore
from ..errors import AnalystError


def add_parser(sub: argparse._SubParsersAction) -> None:
    m = sub.add_parser("model", help="Round win probability model and swing ratings (free: no API calls).")
    msub = m.add_subparsers(dest="model_command", required=True)
    t = msub.add_parser("train", help="Train on every parsed demo, evaluated leave-one-demo-out.")
    t.add_argument("--ingest", action="store_true", help="First parse every .dem in CS2_DEMO_DIR that isn't cached.")
    s = msub.add_parser("swing", help="Swing ratings for one demo, or the win probability timeline of one round.")
    s.add_argument("demo", help="A .dem path, a file name in CS2_DEMO_DIR, or a cached demo_id.")
    s.add_argument("--round", type=int, help="Show this round's win probability step by step.")


def model_dir(settings) -> Path:
    return settings.cache_dir / "winprob"


def all_states(store: DemoStore, console: Console) -> pl.DataFrame:
    from .features import states_for

    frames = []
    for d in store.list_demos():
        try:
            frames.append(states_for(store, d["demo_id"]))
        except Exception as e:  # one odd demo shouldn't stop training
            console.print(f"[yellow]skipped {d['demo_id']} ({d.get('map')}): {e}[/]")
    frames = [f for f in frames if f.height]
    if not frames:
        sys.exit("No parsed demos. Ingest some first (cs2-agent model train --ingest, or run the evals).")
    return pl.concat(frames, how="diagonal_relaxed")


def resolve_demo(store: DemoStore, settings, demo: str) -> str:
    known = {d["demo_id"] for d in store.list_demos()}
    if demo in known:
        return demo
    p = Path(demo).expanduser()
    if not p.is_absolute() and not p.exists() and settings.demo_dir:
        p = settings.demo_dir / p
    if not p.is_file():
        sys.exit(f"No demo '{demo}': not a cached demo_id, and no file at {p}.")
    return store.ingest(p)["demo_id"]


def cmd_train(args: argparse.Namespace, console: Console) -> None:
    from . import model as M

    settings = get_settings()
    store = DemoStore(settings.cache_dir)
    if args.ingest:
        if not settings.demo_dir:
            sys.exit("--ingest needs CS2_DEMO_DIR set in .env.")
        for f in sorted(settings.demo_dir.glob("*.dem")):
            try:
                with console.status(f"Parsing {f.name}..."):
                    meta = store.ingest(f)
                console.print(f"[dim]{f.name}: {meta['map']}, {meta['rounds_played']} rounds"
                              f"{' (cached)' if meta.get('cached') else ''}[/]")
            except AnalystError as e:
                console.print(f"[yellow]skipped {f.name}: {escape(str(e))}[/]")
    refreshed = store.refresh_stale(lambda n: console.print(f"[dim]re-parsing {n} (newer cache format)...[/]"))
    if refreshed:
        console.print(f"[dim]refreshed {len(refreshed)} demos[/]")
    states = all_states(store, console)
    with console.status("Training (leave-one-demo-out)..."):
        try:
            meta = M.train(states, model_dir(settings))
        except ValueError as e:
            sys.exit(str(e))

    console.print(f"Trained on {meta['demos']} demos, {meta['rounds']} rounds, {meta['states']} game states.")
    t = Table(title="Leave-one-demo-out (every score is on a match the model didn't see)", title_justify="left")
    for col in ("Model", "Log loss (lower better)", "Brier (lower better)", "AUC (higher better)"):
        t.add_column(col, justify="right" if col != "Model" else "left")
    names = {"constant": "Always the T win rate", "players_alive": "Players alive only (logistic)",
             "logistic": "Logistic, all features", "xgboost": "XGBoost (monotone), all features",
             "model": "Shipped: average of the two"}
    for k, v in meta["scores"].items():
        t.add_row(names[k], str(v["log_loss"]), str(v["brier"]), "–" if v["auc"] is None else str(v["auc"]))
    console.print(t)
    c = Table(title="Calibration: when the model says X%, how often did T win?", title_justify="left")
    for col in ("Predicted bucket", "States", "Predicted", "Actual"):
        c.add_column(col, justify="right")
    for b in meta["calibration"]:
        c.add_row(b["bucket"], str(b["states"]), f"{b['predicted']:.0%}", f"{b['actual']:.0%}")
    console.print(c)
    imp = Table(title="What drives the estimate (mean |contribution|, log-odds)", title_justify="left")
    imp.add_column("Feature")
    imp.add_column("Importance", justify="right")
    for f, v in meta["importance_mean_abs_contribution"].items():
        imp.add_row(f, str(v))
    console.print(imp)
    console.print(f"[dim]saved to {model_dir(settings)}[/]")


def cmd_swing(args: argparse.Namespace, console: Console) -> None:
    from .. import analysis as A
    from . import model as M
    from . import swing as S
    from .features import states_for

    settings = get_settings()
    store = DemoStore(settings.cache_dir)
    try:
        model, _ = M.load(model_dir(settings))
    except M.ModelMissing as e:
        sys.exit(str(e))
    did = resolve_demo(store, settings, args.demo)
    states = S.with_probabilities(model, states_for(store, did))
    tbl = lambda n: store.table(did, n)  # noqa: E731

    if args.round:
        tl = S.round_timeline(states, model, args.round, store.meta(did)["tick_rate"])
        if not tl:
            sys.exit(f"No round {args.round} in this demo.")
        t = Table(title=f"Round {args.round}: won by {tl['winner_side']}", title_justify="left")
        for col in ("Time", "Event", "Alive T v CT", "P(T wins)", "Swing for T", "Main reasons"):
            t.add_column(col)
        for s in tl["steps"]:
            what = s["event"] if s["event"] == "start" else (
                f"{s['actor']} kills {s['victim']}" if s["event"] == "kill" else f"{s['actor']} {s['event']}s")
            why = ", ".join(w["reason"].replace(" -> ", "→") for w in s["why"][:2])
            t.add_row(f"{s['time_s']}s", escape(what or ""), s["alive"].split(" ")[0], f"{s['p_t_win']:.0%}",
                      f"{s['swing_t_pp']:+.1f}", escape(why))
        console.print(t)
        return

    labels = A.team_labels(tbl("round_players"), tbl("players"))
    ratings = dict(A.with_rating(tbl("stats")).select("steamid", "rating").iter_rows())
    sw = S.swing_ratings(states, tbl("round_players"), labels, ratings)
    t = Table(title="Swing per round (pp) vs rating", title_justify="left")
    for col in ("#", "Player", "Team", "Swing/round", "From kills", "From deaths", "From bomb", "Rating", "Rating #"):
        t.add_column(col)
    for p in sw["players"]:
        t.add_row(str(p["swing_rank"]), escape(p["name"] or ""), escape(p["team"] or ""), f"{p['swing_per_round_pp']:+.2f}",
                  f"{p['from_kills_pp']:+.0f}", f"{p['from_deaths_pp']:+.0f}", f"{p['from_bomb_pp']:+.0f}",
                  str(p["rating"]), str(p.get("rating_rank", "")))
    console.print(t)
    console.print(f"[dim]{sw['note']}[/]")


def run(args: argparse.Namespace) -> None:
    console = Console()
    if args.model_command == "train":
        cmd_train(args, console)
    else:
        cmd_swing(args, console)
