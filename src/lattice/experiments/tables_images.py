"""The image displays: each reduces what its script wrote to the rows the paper shows.

    python -m lattice.experiments.tables <display> --results OUTPUT_ROOT/images --out DIR

``--results`` is the directory the image scripts write to (scripts/_images.sh):
  cifar_reads/      one CSV per CIFAR-10 cell (experiments/read_cifar.py), named by the cell:
                    MODEL_synchronised_nN, MODEL_sequential_nN, MODEL_slopeK_ORDER_nN
  cifar_cells/      MODEL.npz, the sample cells of Figure 2 (experiments/sample_grid.py cells)
  graded_states/    graded_states.npz, the states and level maps of Figure 6
  heldout/          the held-out objective of the mixed model at 60,000 and 120,000 steps
  imagenet_reads/   one CSV per ImageNet-256 cell (experiments/read_imagenet.py), named
                    MODEL_SCHEDULE[_cCALLS]
Every cell is looked up by its name and checked against the model, schedule, slope, order and
sample count its row records, so a raster cell can never stand in for a spread cell. Rows and
columns carry the paper's labels (experiments/tables.py).
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..utils.tables import write_rows

# the CIFAR-10 models, as the rows of Table 7 name them
PIXEL_MODELS = (("iid", "iid levels"), ("mixed", "mixed levels"),
                ("synchronised", "synchronised only"), ("sequential", "sequential only"))
# the same models as the rows of Figure 2 name them
CORNER_MODELS = {"iid": "lattice, iid levels", "mixed": "lattice, mixed levels",
                 "synchronised": "synchronised only", "sequential": "sequential only"}
# the columns of Table 7: (label, schedule, slope, order)
PIXEL_COLUMNS = (("sync.", "synchronised", 0, "raster"), ("slope 1", "sloped", 1, "raster"),
                 ("slope 4 raster", "sloped", 4, "raster"), ("slope 4 spread", "sloped", 4, "spread"),
                 ("slope 16 raster", "sloped", 16, "raster"), ("slope 16 spread", "sloped", 16, "spread"),
                 ("seq.", "sequential", 0, "raster"))
# the pixel columns of Figure 2, all in raster order, and the sample-cell keys of sample_grid cells
CORNER_COLUMNS = (("synchronised", 0, "synchronised"), ("sloped", 1, "sloped1"), ("sloped", 4, "sloped4"),
                  ("sloped", 16, "sloped16"), ("sequential", 0, "sequential"))
FULL = 50000       # samples in every cell of the tables
TREND = (10000, 50000)   # samples in the cells read at twice the training budget


def cell_name(model, schedule, slope, order, n):
    """The file name of a CIFAR-10 cell (scripts/_images.sh names cells the same way)."""
    if schedule == "sloped":
        return f"{model}_slope{slope}_{order}_n{n}"
    return f"{model}_{schedule}_n{n}"


def _row(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run the display's script")
    rows = list(csv.DictReader(open(path)))
    if len(rows) != 1:
        raise SystemExit(f"{path} holds {len(rows)} rows, expected one")
    return rows[0]


def _cell(results, model, schedule, slope, order, n):
    """The row of one CIFAR-10 cell, checked against what the cell is."""
    path = os.path.join(results, "cifar_reads", cell_name(model, schedule, slope, order, n) + ".csv")
    r = _row(path)
    want = dict(model=model, schedule=schedule, slope=str(slope), order=order, samples=str(n))
    got = {k: r[k] for k in want}
    if got != want:
        raise SystemExit(f"{path} records {got}, expected {want}")
    return r


def table07(args, rest):
    """Table 7: CIFAR-10 FID-50K of the four models (rows) at seven schedules (columns), with each
    column's step count, and the mixed model at 120,000 training steps at slope 16 (the row "at 120K
    steps", 50,000 samples); and, in table07_budget.csv, the readings the appendix quotes beside the
    table: the mixed model's held-out objective at 60,000 and 120,000 steps, and its slope-16 cells
    in raster and spread order at 120,000 steps over 10,000 and over 50,000 samples."""
    steps = {"model": "steps"}
    for label, s, k, o in PIXEL_COLUMNS:
        steps[label] = int(_cell(args.results, "iid", s, k, o, FULL)["calls"])
    rows = [steps]
    for model, name in PIXEL_MODELS:
        row = {"model": name}
        for label, s, k, o in PIXEL_COLUMNS:
            c = _cell(args.results, model, s, k, o, FULL)
            if int(c["calls"]) != steps[label]:
                raise SystemExit(f"{model} {label}: {c['calls']} calls, the column has {steps[label]}")
            row[label] = float(c["fid"])
        rows.append(row)
        if model == "mixed":
            row = {"model": "at 120K steps"}
            for label, s, k, o in PIXEL_COLUMNS:
                if (s, k) == ("sloped", 16):
                    c = _cell(args.results, "mixed_120k", s, k, o, FULL)
                    if int(c["calls"]) != steps[label]:
                        raise SystemExit(f"mixed_120k {label}: {c['calls']} calls, the column has {steps[label]}")
                    row[label] = float(c["fid"])
                else:
                    row[label] = ""
            rows.append(row)
    write_rows(rows, args.out, "table07")
    budget = []
    for r in csv.DictReader(open(os.path.join(args.results, "heldout", "heldout_mixed.csv"))):
        budget.append({"model": "mixed levels", "training steps": int(r["step"]), "reading": "held-out objective",
                       "value": float(r["heldout_objective"])})
    for n in TREND:
        for order in ("raster", "spread"):
            c = _cell(args.results, "mixed_120k", "sloped", 16, order, n)
            budget.append({"model": "mixed levels", "training steps": 120000, "reading": f"FID, slope 16, {order}",
                           "samples": n, "value": float(c["fid"])})
    write_rows(budget, args.out, "table07_budget")


def figure02_pixels(args, rest):
    """Figure 2, pixel block: for each model and each raster-order schedule, the FID-50K label (the
    cell of Table 7), its step count, and where its sample images are (the array KEY of
    cifar_cells/MODEL.npz under the results directory; the figure shows the first two)."""
    rows = []
    for model, _ in PIXEL_MODELS:
        cells = os.path.join(args.results, "cifar_cells", f"{model}.npz")
        keys = set(np.load(cells).files)
        for s, k, key in CORNER_COLUMNS:
            c = _cell(args.results, model, s, k, "raster", FULL)
            if key not in keys:
                raise SystemExit(f"{cells} has no samples for {key}")
            rows.append({"model": CORNER_MODELS[model], "schedule": s, "slope": k, "order": "raster",
                         "steps": int(c["calls"]), "FID": float(c["fid"]), "samples": f"cifar_cells/{model}.npz",
                         "key": key})
    write_rows(rows, args.out, "figure02_pixels")


def figure06(args, rest):
    """The numbers of Figure 6: for each schedule and each recorded moment of its walk, the mean
    level index down each patch column (0 noisy, 128 clean, left to right) and the fall from the
    leftmost to the rightmost column; the caption quotes the fourth moment."""
    z = np.load(os.path.join(args.results, "graded_states", "graded_states.npz"))
    level = z["level"].astype(np.float64)              # (rows, moments, 32, 32)
    patch = int(z["patch"])
    rows = []
    for i, schedule in enumerate(z["schedule"]):
        for m, t in enumerate(z["moment"][i]):
            cols = level[i, m].mean(axis=0)            # the mean level down each image column
            means = cols.reshape(-1, patch).mean(axis=1)
            row = {"schedule": str(schedule), "slope": int(z["slope"][i]), "order": str(z["order"][i]),
                   "steps": int(z["calls"][i]), "moment": m + 1, "path index": int(t)}
            row.update({f"column {j + 1}": float(v) for j, v in enumerate(means)})
            row["left minus right"] = float(cols[0] - cols[-1])
            rows.append(row)
    write_rows(rows, args.out, "figure06")


# the rows of Table 8: (row label, model, schedule, calls or None for the corner's full path)
IMAGENET_CELLS = (("lattice, mixed levels", "lattice", "synchronised", None),
                  ("lattice, mixed levels", "lattice", "sequential", None),
                  ("synchronised only", "synchronised", "synchronised", None),
                  ("synchronised only", "synchronised", "sequential", None),
                  ("lattice, synchronised at 16 steps", "lattice", "synchronised", 16))
IMAGENET_MEASURE = {"lattice": "mixed", "synchronised": "synchronised"}   # each model's training distribution
IMAGENET_CALLS = {"synchronised": 249, "sequential": 256}                 # each corner's full path


def imagenet_cell_name(model, schedule, calls=None):
    """The file name of an ImageNet-256 cell (scripts/_images.sh names cells the same way)."""
    return f"{model}_{schedule}" + (f"_c{calls}" if calls else "")


def table08(args, rest):
    """Table 8: ImageNet-256 FID-50K (with sFID, Inception Score, precision and recall) of the two
    SiT-B/2 models at the synchronised corner (249 calls) and the sequential corner (256 calls),
    and the lattice model at the synchronised corner with 16 calls."""
    rows = []
    for label, model, schedule, calls in IMAGENET_CELLS:
        path = os.path.join(args.results, "imagenet_reads", imagenet_cell_name(model, schedule, calls) + ".csv")
        r = _row(path)
        want = dict(measure=IMAGENET_MEASURE[model], schedule=schedule, calls=str(calls or IMAGENET_CALLS[schedule]),
                    samples=str(FULL))
        got = {k: r[k] for k in want}
        if got != want:
            raise SystemExit(f"{path} records {got}, expected {want}")
        rows.append({"model": label, "corner": schedule, "steps": int(r["calls"]), "FID": float(r["fid"]),
                     "sFID": float(r["sfid"]), "Inception Score": float(r["inception_score"]),
                     "precision": float(r["precision"]), "recall": float(r["recall"])})
    write_rows(rows, args.out, "table08")


DISPLAYS = {f.__name__: f for f in (figure02_pixels, figure06, table07, table08)}
