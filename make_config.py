"""
Generate a configs.txt file for the sbatch array driver.

Each line is a complete set of CLI flags for `survey_propagation.py`.
You can edit / pipe / filter freely; this is just a starting point.

Example:
    python make_configs.py \\
        --K 2 3 4 --d-range 3 12 --modes assortative disassortative \\
        --H-strategy near-half --extra "--gamma 0.15 --max-iter 500"

Strategies for picking H given d:
    all       : H in 0..d
    near-half : H in {ceil(d/K) - 1, ceil(d/K), ..., ceil(d/K) + delta}
    custom    : provide --H values explicitly

Usage: python make_configs.py [options] > configs.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable


def _h_values(strategy: str, d: int, K: int, H_list, delta: int) -> Iterable[int]:
    if strategy == "all":
        return range(0, d + 1)
    if strategy == "custom":
        return [h for h in H_list if 0 <= h <= d]
    if strategy == "near-half":
        center = math.ceil(d / K)
        lo = max(0, center - 1)
        hi = min(d, center + delta)
        return range(lo, hi + 1)
    raise ValueError(f"Unknown strategy {strategy!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--d-range", type=int, nargs=2, default=[3, 12],
                   metavar=("MIN", "MAX"))
    p.add_argument("--modes", nargs="+",
                   default=["assortative", "disassortative"])
    p.add_argument("--H-strategy", choices=("all", "near-half", "custom"),
                   default="near-half")
    p.add_argument("--H", type=int, nargs="*", default=[],
                   help="Used when --H-strategy custom.")
    p.add_argument("--delta", type=int, default=3,
                   help="Span around ceil(d/K) for near-half strategy.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", default="",
                   help="Extra CLI flags appended to every config line.")
    args = p.parse_args()

    d_min, d_max = args.d_range
    extra = (" " + args.extra) if args.extra else ""

    for K in args.K:
        for d in range(d_min, d_max + 1):
            for H in _h_values(args.H_strategy, d, K, args.H, args.delta):
                for mode in args.modes:
                    print(
                        f"--K {K} --d {d} --H {H} --mode {mode} "
                        f"--seed {args.seed}{extra}"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())