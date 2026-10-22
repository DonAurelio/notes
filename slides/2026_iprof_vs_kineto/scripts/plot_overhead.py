#!/usr/bin/env python3
"""Plot overhead summary metrics from summary.csv → analysis/overhead.png"""

import argparse, csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO = Path(__file__).parent.parent

_ap = argparse.ArgumentParser()
_ap.add_argument("--summary", default=str(REPO / "results" / "summary.csv"))
_ap.add_argument("--out",     default=str(REPO / "analysis" / "overhead.png"))
_args, _ = _ap.parse_known_args()

SUMMARY = Path(_args.summary)
OUTFILE = Path(_args.out)

SIZES = [256, 1024, 4096]
ITERS = [10, 100, 1000]
CELLS = [(s, n) for s in SIZES for n in ITERS if (s, n) != (4096, 1000)]

# Colorblind-safe (Wong 2011)
COLOR = {"IP-XPU": "#0072B2", "KN-XPU": "#E69F00"}
LS_S  = {256: "-",  1024: "--", 4096: ":"}
MK_S  = {256: "o",  1024: "s",  4096: "^"}
LS_N  = {10:  "-",  100:  "--", 1000: ":"}
MK_N  = {10:  "o",  100:  "s",  1000: "^"}

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7,
    "legend.framealpha":  0.7,
})

def load():
    d = {}
    with open(SUMMARY) as f:
        for row in csv.DictReader(f):
            key = (row["scenario"], int(row["S"]), int(row["N"]))
            d[key] = {k: float(v) if v else None for k, v in row.items()
                      if k not in ("scenario", "S", "N")}
    return d

def get(d, sc, s, n, field):
    return (d.get((sc, s, n)) or {}).get(field)

def main():
    if not SUMMARY.exists():
        sys.exit(f"ERROR: {SUMMARY} not found — run summarize.py first")

    d = load()
    OUTFILE.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("iprof vs Kineto — XPU Tracing Overhead", fontsize=13, fontweight="bold", y=0.98)

    ax_ratio, ax_iter, ax_start, ax_ape = axes.flat
    GRID = dict(axis="y", color="#eeeeee", lw=0.8, zorder=0)

    # ── A: overhead_ratio [%] vs N — shows amortization ───────────────────────
    for sc in ["IP-XPU", "KN-XPU"]:
        for s in SIZES:
            xs = [n for n in ITERS if get(d, sc, s, n, "overhead_ratio_pct") is not None]
            ys = [get(d, sc, s, n, "overhead_ratio_pct") for n in xs]
            if not xs:
                continue
            ax_ratio.plot(xs, ys, color=COLOR[sc], ls=LS_S[s], marker=MK_S[s],
                          ms=7, lw=1.5, label=f"{sc} S={s}")

    ax_ratio.set_xscale("log")
    ax_ratio.set_xticks(ITERS)
    ax_ratio.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_ratio.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax_ratio.set_xlabel("N (iterations)")
    ax_ratio.set_ylabel("overhead_ratio [%]")
    ax_ratio.set_title("A · Hot-region overhead — amortization as N grows")
    ax_ratio.legend(ncol=2)
    ax_ratio.grid(**GRID)

    # ── B: overhead_per_iter [ms/iter] vs S — shows compute scaling ────────────
    for sc in ["IP-XPU", "KN-XPU"]:
        for n in ITERS:
            xs = [s for s in SIZES if get(d, sc, s, n, "overhead_per_iter_ms") is not None]
            ys = [get(d, sc, s, n, "overhead_per_iter_ms") for s in xs]
            if not xs:
                continue
            ax_iter.plot(xs, ys, color=COLOR[sc], ls=LS_N[n], marker=MK_N[n],
                         ms=7, lw=1.5, label=f"{sc} N={n}")

    ax_iter.set_xscale("log")
    ax_iter.set_yscale("log")
    ax_iter.set_xticks(SIZES)
    ax_iter.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_iter.set_xlabel("S (matrix side length)")
    ax_iter.set_ylabel("overhead_per_iter [ms/iter]")
    ax_iter.set_title("B · Per-iteration overhead vs matrix size (log–log)")
    ax_iter.legend(ncol=2)
    ax_iter.grid(axis="both", color="#eeeeee", lw=0.8, zorder=0)

    # ── C: startup_overhead [s] — expect ~7.5s for IP, ~0 for KN ──────────────
    cell_labels = [f"S={s}\nN={n}" for s, n in CELLS]
    y = range(len(CELLS))
    bar_h = 0.35

    for i, sc in enumerate(["IP-XPU", "KN-XPU"]):
        vals = [get(d, sc, s, n, "startup_overhead_s") or 0 for s, n in CELLS]
        ax_start.barh([yi + (i - 0.5) * bar_h for yi in y], vals,
                      height=bar_h, color=COLOR[sc], label=sc, alpha=0.9, zorder=3)

    ax_start.axvline(7.5, color="#888888", ls="--", lw=1, label="7.5 s pilot ref")
    ax_start.set_yticks(list(y))
    ax_start.set_yticklabels(cell_labels, fontsize=7)
    ax_start.set_xlabel("startup_overhead [s]")
    ax_start.set_title("C · Fixed startup overhead per cell")
    ax_start.legend()
    ax_start.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)

    # ── D: analysis_per_event [ns/event] — log scale, expected iprof >> Kineto ─
    for i, sc in enumerate(["IP-XPU", "KN-XPU"]):
        vals = [get(d, sc, s, n, "analysis_per_event_ns") or 0 for s, n in CELLS]
        ax_ape.barh([yi + (i - 0.5) * bar_h for yi in y], vals,
                    height=bar_h, color=COLOR[sc], label=sc, alpha=0.9, zorder=3)

    ax_ape.set_xscale("log")
    ax_ape.set_yticks(list(y))
    ax_ape.set_yticklabels(cell_labels, fontsize=7)
    ax_ape.set_xlabel("analysis_per_event [ns/event]  (log scale)")
    ax_ape.set_title("D · Analysis cost per event")
    ax_ape.legend()
    ax_ape.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)

    plt.tight_layout()
    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTFILE}")

if __name__ == "__main__":
    main()
