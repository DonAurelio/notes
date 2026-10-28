#!/usr/bin/env python3
"""Visualizations for the DDP iprof vs Kineto experiment (same style as 2026_iprof_vs_kineto)."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

EXP  = Path(__file__).parent.parent
RAW  = EXP / "results" / "raw.csv"
VIZ  = EXP / "viz"
VIZ.mkdir(exist_ok=True)

BLUE   = "#2a78d6"
ORANGE = "#eb6834"

df = pd.read_csv(RAW)
for col in ["T_hot_s", "T_process_s", "n_events", "analysis_s"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

bl = df[df.scenario == "BL"]["T_hot_s"].mean()
ip = df[df.scenario == "IP"]["T_hot_s"].mean()
kn = df[df.scenario == "KN"]["T_hot_s"].mean()

# ── 1. overhead_ratio.png ─────────────────────────────────────────────────────
ip_ov = (ip - bl) / bl * 100
kn_ov = (kn - bl) / bl * 100

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["iprof", "Kineto"], [ip_ov, kn_ov], color=[BLUE, ORANGE], width=0.4, zorder=3)
ax.axhline(0, color="#c3c2b7", linewidth=1)
for bar, val in zip(bars, [ip_ov, kn_ov]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + (1 if val >= 0 else -3),
        f"{val:+.1f}%",
        ha="center", va="bottom" if val >= 0 else "top", fontsize=9,
    )
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("T_hot overhead vs baseline")
ax.set_title("Tracing overhead — T_hot relative to baseline\n(DDP, TinyLlama, 2 nodes)")
ax.grid(axis="y", linewidth=0.5, color="#e1e0d9", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(VIZ / "overhead_ratio.png", dpi=150)
plt.close(fig)
print("Wrote overhead_ratio.png")

# ── 2. n_events_comparison.png ────────────────────────────────────────────────
ip_ev = df[df.scenario == "IP"]["n_events"].mean()
kn_ev = df[df.scenario == "KN"]["n_events"].mean()

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["iprof", "Kineto"], [ip_ev, kn_ev], color=[BLUE, ORANGE], width=0.4, zorder=3)
for x, val in enumerate([ip_ev, kn_ev]):
    ax.text(x, val + max(ip_ev, kn_ev) * 0.01, f"{int(val):,}", ha="center", va="bottom", fontsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
ax.set_ylabel("n_events (raw trace records)")
ax.set_title("Event count by tool\n(DDP, TinyLlama, 2 nodes)")
ax.grid(axis="y", linewidth=0.5, color="#e1e0d9", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(VIZ / "n_events_comparison.png", dpi=150)
plt.close(fig)
print("Wrote n_events_comparison.png")

# ── 3. analysis_per_event.png ─────────────────────────────────────────────────
ip_sub = df[df.scenario == "IP"][["analysis_s", "n_events"]].mean()
kn_sub = df[df.scenario == "KN"][["analysis_s", "n_events"]].mean()

ip_ape = ip_sub["analysis_s"] / ip_sub["n_events"] * 1e9
kn_ape = kn_sub["analysis_s"] / kn_sub["n_events"] * 1e9

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["iprof", "Kineto"], [ip_ape, kn_ape], color=[BLUE, ORANGE], width=0.4, zorder=3)
for x, val in enumerate([ip_ape, kn_ape]):
    ax.text(x, val + max(ip_ape, kn_ape) * 0.01, f"{val:.0f} ns", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("analysis time / n_events  (ns/event)")
ax.set_title("Analysis cost per event\n(DDP, TinyLlama, 2 nodes)")
ax.grid(axis="y", linewidth=0.5, color="#e1e0d9", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(VIZ / "analysis_per_event.png", dpi=150)
plt.close(fig)
print("Wrote analysis_per_event.png")
