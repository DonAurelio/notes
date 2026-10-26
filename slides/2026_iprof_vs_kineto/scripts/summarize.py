#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent.parent
RAW  = REPO / "results" / "raw.csv"

df = pd.read_csv(RAW)

# Table 1: T_hot mean ± std per scenario, pivoted wide (one row per S, N)
thot = (df.groupby(["scenario", "S", "N"])["T_hot_s"]
          .agg(mean="mean", std="std")
          .unstack("scenario"))
thot.columns = [f"{sc.replace('-XPU', '')}_{stat}" for stat, sc in thot.columns]
thot.reset_index().to_csv(REPO / "results" / "summary_thot.csv", index=False)

# Table 2: all timing metrics mean ± std, one row per (scenario, S, N)
timing = (df.groupby(["scenario", "S", "N"])[["T_hot_s", "T_process_s", "analysis_s"]]
            .agg(["mean", "std"])
            .round(6))
timing.columns = ["_".join(c) for c in timing.columns]
timing.reset_index().to_csv(REPO / "results" / "summary_timing.csv", index=False)

print("Wrote", REPO / "results" / "summary_thot.csv")
print("Wrote", REPO / "results" / "summary_timing.csv")
