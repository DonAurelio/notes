#!/usr/bin/env python3
"""Validates raw.csv completeness and flags missing or incomplete rows."""

import csv, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).parent.parent
RAW  = REPO / "results" / "raw.csv"

SCENARIOS = ["BL-XPU", "IP-XPU", "KN-XPU"]
SIZES     = [256, 1024, 4096]
ITERS     = [10, 100, 1000]
REPS      = 15
EXCLUDE   = {(4096, 1000)}

CELLS    = [(s, n) for s in SIZES for n in ITERS if (s, n) not in EXCLUDE]
EXPECTED = {(sc, s, n, r)
            for sc in SCENARIOS for s, n in CELLS for r in range(1, REPS + 1)}

def main():
    if not RAW.exists():
        sys.exit(f"ERROR: {RAW} not found")

    seen, issues = set(), []

    with open(RAW) as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            key = (row["scenario"], int(row["S"]), int(row["N"]), int(row["rep"]))

            if key in seen:
                issues.append(f"line {i}: duplicate {key}")
            seen.add(key)

            if not row["T_hot_s"] or not row["T_process_s"]:
                issues.append(f"line {i}: missing timing  {key}")

            if row["scenario"] == "IP-XPU" and (not row["n_events"] or not row["analysis_s"]):
                issues.append(f"line {i}: IP-XPU missing replay data  {key}")

    missing = sorted(EXPECTED - seen)

    print(f"Rows found:   {len(seen)}")
    print(f"Expected:     {len(EXPECTED)}")
    print(f"Missing:      {len(missing)}")
    for sc, s, n, r in missing:
        print(f"  MISSING  {sc}  S={s}  N={n}  rep={r}")
    if issues:
        print(f"\nIssues ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("No issues.")

    if missing or issues:
        sys.exit(1)
    print("\nraw.csv is complete and valid.")

if __name__ == "__main__":
    main()
