"""Reduce the experiments' outputs to the numbers a table or figure of the paper shows.

    python -m lattice.experiments.tables <display> --out DIR [--results DIR] [...]

A display is named after the table or figure it reduces: table05 is Table 5 and figure06 is
Figure 6. A table or figure whose parts come from different experiments has one display per part:
table03_text8 and table03_mar are the left and right halves of Table 3, figure05_text8 and
figure05_mar the two panels of Figure 5. Each display reads the files its script produced, prints
the rows and writes them to OUT/<display>.csv, where the scripts pass $OUTPUT_ROOT/tableNN or
$OUTPUT_ROOT/figureNN as OUT. The rows and columns carry the paper's labels. A cell that prints
two numbers takes two columns: a value and its spread (a column named with sd, se or s.e.m.), a
value and its interval ("95% low", "95% high"), or an estimate and a measurement. A step budget
is a column "N steps" or the value of a column "steps". A reading that the caption or the text
quotes beside a table, and that is not a row of it, goes to OUT/<display>_<part>.csv. The
displays live beside the experiments they reduce, one module per family (tables_<family>.py, each
with a DISPLAYS dict); this module only collects them.
"""
from __future__ import annotations

import argparse
import importlib

FAMILIES = ("text8", "released", "images")


def displays():
    out = {}
    for family in FAMILIES:
        module = importlib.import_module(f"{__package__}.tables_{family}")
        clash = set(out) & set(module.DISPLAYS)
        if clash:
            raise RuntimeError(f"displays defined twice: {sorted(clash)}")
        out.update(module.DISPLAYS)
    return out


def main(argv=None):
    table = displays()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("display", choices=sorted(table))
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="", help="the directory the display's experiments wrote to")
    args, rest = ap.parse_known_args(argv)
    table[args.display](args, rest)


if __name__ == "__main__":
    main()
