#!/usr/bin/env python3
import argparse
import os
import re

from frameopt.core.energy import coherence
from frameopt.core.frame import Frame

FNAME_RE = re.compile(r"^(\d+)x(\d+).*.txt$", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser(
        description="Compute coherence of a frame saved as txt."
    )
    ap.add_argument(
        "frame_file", type=str, help="Path to file named like 'dxn_init.txt'"
    )
    args = ap.parse_args()

    fname = os.path.basename(args.frame_file)
    m = FNAME_RE.search(fname)
    if not m:
        raise ValueError(
            "Filename must start with 'dxn' and end with .txt (e.g. '4x6.txt', '4x6_jrr.txt')."
        )
    d, n = map(int, m.groups())

    frame = Frame.load_txt(args.frame_file, n=n, d=d)
    val = coherence(frame)
    print(val)


if __name__ == "__main__":
    main()
