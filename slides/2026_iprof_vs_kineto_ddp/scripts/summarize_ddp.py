#!/usr/bin/env python3
"""Summarize raw.csv into mean ± std tables for the DDP iprof vs Kineto experiment."""

import pandas as pd
from pathlib import Path

EXP = Path(__file__).parent.parent
RESULTS = EXP / "results"
RAW = RESULTS / "raw.csv"

df = pd.read_csv(RAW)

metrics = ["T_hot_s", "T_process_s", "n_events", "analysis_s"]
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# summary_thot.csv: one row per scenario with mean/std of T_hot_s
thot = (
    df.groupby("scenario")["T_hot_s"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "T_hot_mean", "std": "T_hot_std"})
    .reset_index()
)
thot.to_csv(RESULTS / "summary_thot.csv", index=False)
print("Wrote summary_thot.csv")
print(thot.to_string(index=False))

# summary_timing.csv: mean/std for all metrics per scenario
timing = (
    df.groupby("scenario")[metrics]
    .agg(["mean", "std"])
)
timing.columns = ["_".join(c) for c in timing.columns]
timing = timing.reset_index()
timing.to_csv(RESULTS / "summary_timing.csv", index=False)
print("\nWrote summary_timing.csv")
print(timing.to_string(index=False))
