#!/usr/bin/env python3
import argparse
import re

from frameopt.core.energy import coherence
from frameopt.core.frame import Frame

FNAME_RE = re.compile(r"(\d+)x(\d+)_\w{3,4}\.txt$")


def main():
    ap = argparse.ArgumentParser(
        description="Compute coherence of a frame saved as txt."
    )
    ap.add_argument(
        "frame_file", type=str, help="Path to file named like 'dxn_init.txt'"
    )
    args = ap.parse_args()

    m = FNAME_RE.search(args.frame_file)
    if not m:
        raise ValueError(
            "Filename must be of form 'dxn_<tag>.txt' (e.g. '4x6_jrr.txt')."
        )
    d, n = map(int, m.groups())

    frame = Frame.load_txt(args.frame_file, n=n, d=d)
    val = coherence(frame)
    print(val)


if __name__ == "__main__":
    main()
