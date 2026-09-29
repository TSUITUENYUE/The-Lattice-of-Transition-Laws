"""Writing a display's rows: one CSV per display, and the rows printed as they are written."""
from __future__ import annotations

import csv
import os


def write_rows(rows, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"wrote {path}")
