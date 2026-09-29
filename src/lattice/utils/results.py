"""Where a number goes after it is measured.

Every entry point writes its rows to one CSV in the directory given by --out and prints each row
as it arrives, so a table has exactly one file behind it and a log can be checked against that
file. The header is the union of the keys the rows carry, in the order they first appear, and a
missing cell is left empty.
"""
from __future__ import annotations

import csv
from pathlib import Path


class Rows:
    """Accumulate rows, print each as it arrives, write once at the end (or on leaving a with)."""

    def __init__(self, name, out_dir, meta=None, seed=None):
        self.name = name
        self.seed = seed
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if meta:
            print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in meta.items()), flush=True)

    def add(self, **row):
        self.rows.append(row)
        bits = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(f"ROW {self.name} {bits}", flush=True)
        return row

    @property
    def path(self):
        stem = self.name if self.seed is None else f"{self.name}_s{self.seed}"
        return self.dir / f"{stem}.csv"

    def write(self):
        if not self.rows:
            return None
        keys = []
        for row in self.rows:
            keys += [k for k in row if k not in keys]
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(self.rows)
        print(f"WROTE {self.path} ({len(self.rows)} rows)", flush=True)
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.write()
        return False
