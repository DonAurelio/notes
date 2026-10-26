#!/usr/bin/env python3
"""Aggregates raw.csv into per-cell summary metrics (median + IQR)."""

import argparse, csv, sys, statistics
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).parent.parent

_ap = argparse.ArgumentParser()
_ap.add_argument("--raw", default=str(REPO / "results" / "raw.csv"))
_ap.add_argument("--out", default=str(REPO / "results" / "summary.csv"))
_args, _ = _ap.parse_known_args()

RAW     = Path(_args.raw)
SUMMARY = Path(_args.out)

SIZES   = [256, 1024, 4096]
ITERS   = [10, 100, 1000]
EXCLUDE = {(4096, 1000)}
TRACED  = ["IP-XPU", "KN-XPU"]

def iqr(vals):
    s = sorted(vals)
    n = len(s)
    if n < 4:
        return None
    q1 = statistics.median(s[:n // 2])
    q3 = statistics.median(s[(n + 1) // 2:])
    return q3 - q1

def agg(rows, field):
    vals = [r[field] for r in rows if r[field] is not None]
    if not vals:
        return None, None
    return statistics.median(vals), iqr(vals)

def load():
    data = defaultdict(list)
    with open(RAW) as f:
        for row in csv.DictReader(f):
            key = (row["scenario"], int(row["S"]), int(row["N"]))
            data[key].append({
                "T_hot_s":     float(row["T_hot_s"]),
                "T_process_s": float(row["T_process_s"]),
                "n_events":    int(row["n_events"])      if row["n_events"]   else None,
                "analysis_s":  float(row["analysis_s"])  if row["analysis_s"] else None,
            })
    return data

def fmt(v, spec):
    return format(v, spec) if v is not None else ""

def main():
    if not RAW.exists():
        sys.exit(f"ERROR: {RAW} not found")

    data = load()

    med = {}
    for (sc, s, n), rows in data.items():
        med[(sc, s, n)] = {
            "T_hot":    agg(rows, "T_hot_s"),
            "T_proc":   agg(rows, "T_process_s"),
            "n_ev":     agg(rows, "n_events"),
            "analysis": agg(rows, "analysis_s"),
        }

    out = []
    for s in SIZES:
        for n in ITERS:
            if (s, n) in EXCLUDE:
                continue
            bl = med.get(("BL-XPU", s, n))
            if bl is None:
                continue
            bl_hot,  _ = bl["T_hot"]
            bl_proc, _ = bl["T_proc"]

            for sc in TRACED:
                tr = med.get((sc, s, n))
                if tr is None:
                    continue
                tr_hot,  tr_hot_iqr = tr["T_hot"]
                tr_proc, _          = tr["T_proc"]
                n_ev,    _          = tr["n_ev"]
                t_an,    t_an_iqr   = tr["analysis"]

                dT_hot  = tr_hot  - bl_hot
                dT_proc = tr_proc - bl_proc

                overhead_ratio    = dT_hot / bl_hot * 100 if bl_hot else None
                overhead_per_iter = dT_hot / n * 1000      # ms/iter
                startup_overhead  = dT_proc - dT_hot       # s
                ape = t_an / n_ev * 1e9 if (t_an and n_ev) else None  # ns/event

                out.append({
                    "scenario":              sc,
                    "S":                     s,
                    "N":                     n,
                    "overhead_ratio_pct":    fmt(overhead_ratio,    ".2f"),
                    "overhead_per_iter_ms":  fmt(overhead_per_iter, ".4f"),
                    "startup_overhead_s":    fmt(startup_overhead,  ".3f"),
                    "T_hot_s_iqr":           fmt(tr_hot_iqr,        ".6f"),
                    "n_events_med":          fmt(n_ev,              ".0f"),
                    "analysis_s_med":        fmt(t_an,              ".3f"),
                    "analysis_s_iqr":        fmt(t_an_iqr,          ".3f"),
                    "analysis_per_event_ns": fmt(ape,               ".1f"),
                })

    fields = [
        "scenario", "S", "N",
        "overhead_ratio_pct", "overhead_per_iter_ms", "startup_overhead_s",
        "T_hot_s_iqr",
        "n_events_med", "analysis_s_med", "analysis_s_iqr", "analysis_per_event_ns",
    ]
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print(f"Wrote {len(out)} rows to {SUMMARY}")

if __name__ == "__main__":
    main()
