#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent
RAW  = REPO / "results" / "raw.csv"
VIZ  = REPO / "viz"
VIZ.mkdir(exist_ok=True)

BLUE   = "#2a78d6"
ORANGE = "#eb6834"
SIZES  = [256, 1024, 4096]
ITERS  = [10, 100, 1000]

df = pd.read_csv(RAW)

# ── helpers ──────────────────────────────────────────────────────────────────

def mean_std(df, scenario, col):
    g = df[df.scenario == scenario].groupby(["S", "N"])[col]
    return g.mean().reset_index(), g.std().reset_index()

def x_positions(n_groups, n_bars, width=0.35):
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * width
    return x, offsets

# ── 1. overhead_ratio.png ─────────────────────────────────────────────────────
# T_hot overhead % vs baseline, grouped bar per (S, N)

bl = df[df.scenario == "BL-XPU"].groupby(["S", "N"])["T_hot_s"].mean()
ip = df[df.scenario == "IP-XPU"].groupby(["S", "N"])["T_hot_s"].mean()
kn = df[df.scenario == "KN-XPU"].groupby(["S", "N"])["T_hot_s"].mean()

labels = [f"S={s}\nN={n}" for s in SIZES for n in ITERS]
ip_ov  = [(ip.get((s, n), np.nan) - bl.get((s, n), np.nan)) / bl.get((s, n), np.nan) * 100
          for s in SIZES for n in ITERS]
kn_ov  = [(kn.get((s, n), np.nan) - bl.get((s, n), np.nan)) / bl.get((s, n), np.nan) * 100
          for s in SIZES for n in ITERS]

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x - w/2, ip_ov, w, label="iprof",  color=BLUE,   zorder=3)
ax.bar(x + w/2, kn_ov, w, label="Kineto", color=ORANGE, zorder=3)
ax.axhline(0, color="#c3c2b7", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("T_hot overhead vs baseline")
ax.set_title("Tracing overhead — T_hot relative to baseline")
ax.legend(frameon=False)
ax.grid(axis="y", linewidth=0.5, color="#e1e0d9", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(VIZ / "overhead_ratio.png", dpi=150)
plt.close(fig)
print("Wrote overhead_ratio.png")

# ── 2. analysis_per_event.png ─────────────────────────────────────────────────
# analysis time / n_events (ns/event) — line per tool, x=N, facet by S

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, S in zip(axes, SIZES):
    for scenario, label, color in [("IP-XPU","iprof",BLUE), ("KN-XPU","Kineto",ORANGE)]:
        sub = df[(df.scenario == scenario) & (df.S == S)].groupby("N")[["analysis_s","n_events"]].mean()
        sub = sub.reindex(ITERS)
        ape = sub["analysis_s"] / sub["n_events"] * 1e9
        valid = ape.dropna()
        ax.plot(valid.index, valid.values, "o-", color=color, linewidth=2,
                markersize=6, label=label)

    ax.set_title(f"S = {S}", fontsize=10)
    ax.set_xlabel("N (iterations)")
    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([str(n) for n in ITERS])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
    ax.grid(linewidth=0.5, color="#e1e0d9")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("analysis time / n_events  (ns/event)")
axes[0].legend(frameon=False)
fig.suptitle("Analysis cost per event", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(VIZ / "analysis_per_event.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote analysis_per_event.png")

# ── 3. n_events_comparison.png ────────────────────────────────────────────────
# n_events mean — line per tool, x=N, facet by S

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, S in zip(axes, SIZES):
    for scenario, label, color in [("IP-XPU","iprof",BLUE), ("KN-XPU","Kineto",ORANGE)]:
        sub = df[(df.scenario == scenario) & (df.S == S)].groupby("N")["n_events"].mean()
        sub = sub.reindex(ITERS).dropna()
        ax.plot(sub.index, sub.values, "o-", color=color, linewidth=2,
                markersize=6, label=label)

    ax.set_title(f"S = {S}", fontsize=10)
    ax.set_xlabel("N (iterations)")
    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([str(n) for n in ITERS])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
    ax.grid(linewidth=0.5, color="#e1e0d9")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("n_events (raw trace records)")
axes[0].legend(frameon=False)
fig.suptitle("Event count by tool and workload", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(VIZ / "n_events_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote n_events_comparison.png")
