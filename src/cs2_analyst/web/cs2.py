"""'Watch in CS2': put the demo where CS2 can play it, and give the console commands to jump to a moment.

A web page can't start or control another program, so the app copies the demo into CS2's folder and
hands back two commands to paste into the in-game console:
    playdemo cs2-analyst/<name>      load the demo
    demo_gototick <tick>             jump a few seconds before the moment
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from ..errors import AnalystError

SUBDIR = "cs2-analyst"
LEAD_IN_S = 5.0
DEFAULT_DIRS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo",
    r"C:\Program Files\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo",
    "~/.steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo",
    "~/.local/share/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo",
]


def csgo_dir() -> Path | None:
    """CS2's game/csgo folder: CS2_GAME_DIR if set, else the usual Steam install locations."""
    env = os.getenv("CS2_GAME_DIR")
    for candidate in ([env] if env else []) + DEFAULT_DIRS:
        p = Path(candidate).expanduser()
        if p.is_dir():
            return p
    return None


def demo_name(source: Path) -> str:
    """A console-safe file name (no spaces or odd characters) for the copy."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", source.stem)


def prepared_path(source: Path, game_dir: Path) -> Path:
    return game_dir / SUBDIR / f"{demo_name(source)}.dem"


def prepare(source: Path, game_dir: Path | None) -> Path:
    if game_dir is None:
        raise AnalystError("Couldn't find CS2's game\\csgo folder. Set CS2_GAME_DIR in .env to it, e.g. "
                           "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\csgo")
    if not source.is_file():
        raise AnalystError(f"The original demo isn't at '{source}' anymore.")
    target = prepared_path(source, game_dir)
    if not (target.exists() and target.stat().st_size == source.stat().st_size):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def commands(source: Path, tick: int | None, tick_rate: float) -> list[str]:
    cmds = [f"playdemo {SUBDIR}/{demo_name(source)}"]
    if tick is not None:
        cmds.append(f"demo_gototick {max(0, int(tick - LEAD_IN_S * tick_rate))}")
    return cmds
