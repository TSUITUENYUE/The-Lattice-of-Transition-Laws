"""Records written whole: one JSON file per decoded item, and the settings a directory was made with.

An interrupted run leaves either a complete record or none, so running it again resumes at the first
item without one. A directory's protocol.json holds the settings its records were made with, and a run
with other settings refuses to add to it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_json(path, value):
    """Write value as JSON to path through a temporary file and a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def check_protocol(directory, protocol):
    """Record the settings of a directory, or check that they are the ones it was made with."""
    path = Path(directory) / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != protocol:
            changed = sorted(k for k in set(old) | set(protocol) if old.get(k) != protocol.get(k))
            raise SystemExit(f"{directory} was made with other settings ({', '.join(changed)}); "
                             f"use a new output directory")
    else:
        atomic_json(path, protocol)
