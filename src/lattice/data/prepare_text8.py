"""Write text8 as uint8 files on the standard 90M/5M/5M split.

    python -m lattice.data.prepare_text8 --out DATA_ROOT/text8 [--zip text8.zip]

The first 90M characters train, the next 5M validate and the last 5M test. Model selection reads
valid.bin only and every reported number reads test.bin. Characters are coded space -> 0 and
a-z -> 1-26. Downloads text8.zip into --out unless --zip names a copy.
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "http://mattmahoney.net/dc/text8.zip"
SPLIT = (("train", 0, 90_000_000), ("valid", 90_000_000, 95_000_000),
         ("test", 95_000_000, 100_000_000))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for train.bin, valid.bin, test.bin")
    ap.add_argument("--zip", default="", help="an already downloaded text8.zip")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = Path(args.zip) if args.zip else out / "text8.zip"
    if not z.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, z)
    raw = np.frombuffer(zipfile.ZipFile(z).read("text8"), dtype=np.uint8)
    assert len(raw) == 100_000_000, len(raw)
    letters = (raw >= ord("a")) & (raw <= ord("z"))
    assert bool((letters | (raw == ord(" "))).all()), "text8 holds only a-z and spaces"
    codes = np.where(raw == ord(" "), 0, raw - ord("a") + 1).astype(np.uint8)
    for name, lo, hi in SPLIT:
        codes[lo:hi].tofile(out / f"{name}.bin")
        print(f"{name}.bin: {hi - lo} characters", flush=True)


if __name__ == "__main__":
    main()
