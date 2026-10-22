"""One canonical spelling for sides.

awpy reports sides as team numbers (2/3) in some tables and as
"terrorist"/"counter-terrorist" in others. Normalising once at ingest means
analysis code and the model only ever see "CT" or "T".
"""

from __future__ import annotations

_MAP = {
    "2": "T", "t": "T", "terrorist": "T", "terrorists": "T",
    "3": "CT", "ct": "CT", "counter-terrorist": "CT", "counterterrorist": "CT", "counter-terrorists": "CT",
}


def norm_side(value: object) -> str | None:
    if value is None:
        return None
    return _MAP.get(str(value).strip().lower(), str(value))
