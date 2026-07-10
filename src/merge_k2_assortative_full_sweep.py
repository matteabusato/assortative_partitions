#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("results/k2_full_sweep")
SUMMARY_DIR = ROOT / "summaries"
OUTPUT = ROOT / "summary.csv"


def main() -> None:
    paths = sorted(SUMMARY_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON summaries found in {SUMMARY_DIR}")

    rows = []
    for path in paths:
        with path.open() as f:
            rows.append(json.load(f))

    rows.sort(key=lambda row: (int(row["d"]), int(row["H"]), int(row["seed"])))

    fieldnames = list(rows[0].keys())
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(rows)} runs into {OUTPUT}")

    missing = 43 - len(rows)
    if missing:
        print(f"Warning: {missing} of the expected 43 runs are missing.")


if __name__ == "__main__":
    main()
