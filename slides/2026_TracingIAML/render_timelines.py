"""
Render simple per-op Gantt-style timeline PNGs from the iprof (.pftrace) and
torch.profiler/Kineto (chrome-trace .json) traces captured for the
"randn + matmul inside emit_itt()" experiment, for embedding in the
2026_TracingIAML slides.

Run with the frameworks python (matplotlib + perfetto packages installed
there):
    source ~/.claude/skills/pytorch-basic-tracing-environment-config/scripts/env.sh lttng
    python render_timelines.py
"""
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from perfetto.trace_processor import TraceProcessor

IPROF_TRACE = "/home/avivasmeza/dev/pr1_itt/02_randn_matmul_single_scope/iprof_timeline.pftrace"
KINETO_TRACE = "/home/avivasmeza/dev/pr1_itt/02_randn_matmul_single_scope/kineto_timeline.json"
OUT_DIR = "/home/avivasmeza/dev/pr1_itt/notes/slides/2026_TracingIAML"

CPU_COLOR = "#4C78A8"
GPU_COLOR = "#F58518"

# Fixed row order (top to bottom) so both charts line up the same way.
ROW_ORDER = ["aten::randn", "aten::normal_", "aten::matmul", "aten::mm", "gpu_randn_kernel", "gemm_kernel"]
ROW_LABELS = {
    "aten::randn": "aten::randn (CPU)",
    "aten::normal_": "aten::normal_ (CPU)",
    "aten::matmul": "aten::matmul (CPU)",
    "aten::mm": "aten::mm (CPU)",
    "gpu_randn_kernel": "randn fill kernel (GPU)",
    "gemm_kernel": "gemm_kernel (GPU)",
}


def draw_timeline(events, title, out_path):
    """events: dict row_key -> (start_us, dur_us)."""
    rows = [r for r in ROW_ORDER if r in events]
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=160)

    t0 = min(v[0] for v in events.values())
    t_end = max(v[0] + v[1] for v in events.values())
    span = t_end - t0

    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        start, dur = events[row]
        is_gpu = row in ("gemm_kernel", "gpu_randn_kernel")
        color = GPU_COLOR if is_gpu else CPU_COLOR
        x = start - t0
        w = dur
        ax.add_patch(Rectangle((x, y - 0.35), max(w, span * 0.0015), 0.7,
                                facecolor=color, edgecolor="black", linewidth=0.6))
        # Label to the right of the bar so it never gets clipped by a narrow bar.
        label = f"{ROW_LABELS[row]} — {dur:,.1f}µs"
        ax.text(x + max(w, span * 0.0015) + span * 0.012, y, label,
                ha="left", va="center", fontsize=8.5)

    ax.set_xlim(-span * 0.02, span * 1.55)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Time (µs, relative to first event)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Small legend
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=CPU_COLOR, label="CPU (host)"))
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=GPU_COLOR, label="GPU (device)"))
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print("wrote", out_path)


def render_iprof():
    tp = TraceProcessor(trace=IPROF_TRACE)
    q = tp.query("""
        select s.name, s.ts, s.dur, t.name as track_name
        from slice s join track t on s.track_id = t.id
        where s.name in (
            'aten::randn','aten::normal_','aten::matmul','aten::mm','gemm_kernel'
        )
        order by s.ts
    """)
    events = {}
    for row in q:
        # ts/dur are nanoseconds -> microseconds; keep first occurrence per name
        key = row.name
        if key in events:
            continue
        events[key] = (row.ts / 1000.0, row.dur / 1000.0)
    tp.close()
    draw_timeline(events, "THAPI/iprof timeline — randn + matmul inside emit_itt()",
                  f"{OUT_DIR}/iprof_timeline.png")


def render_kineto():
    d = json.load(open(KINETO_TRACE))
    name_to_key = {
        "aten::randn": "aten::randn",
        "aten::normal_": "aten::normal_",
        "aten::matmul": "aten::matmul",
        "aten::mm": "aten::mm",
        "gemm_kernel": "gemm_kernel",
        "at::native::xpu::DistributionElementwiseKernelFunctor<float, float, 4, "
        "at::native::templates::xpu::Normal4DistributionFunctor, "
        "at::native::templates::xpu::NormalTransformFunctor<float, float>, int>": "gpu_randn_kernel",
    }
    events = {}
    for e in d["traceEvents"]:
        if e.get("ph") != "X":
            continue
        key = name_to_key.get(e.get("name", ""))
        if key is None or key in events:
            continue
        events[key] = (e["ts"], e["dur"])
    draw_timeline(events, "torch.profiler/Kineto timeline — randn + matmul",
                  f"{OUT_DIR}/kineto_timeline.png")


def render_vtune():
    # VTune's CLI exposes no per-event timestamp export (no raw trace/timeline
    # dump with start times). These start times were recovered by bisecting
    # `vtune -report hotspots -group-by task|computing-task -time-filter t:END`
    # for the narrowest window that still reports the task's full duration —
    # see the README in this folder for the exact commands.
    # Values are (start_seconds, duration_seconds) from the vtune_xpu_offload result.
    raw = {
        "aten::randn": (207.706601, 0.006225),
        "aten::normal_": (207.707071, 0.005750),
        "aten::matmul": (207.712904, 0.065117),
        "aten::mm": (207.712926, 0.065091),
        "gpu_randn_kernel": (207.712828, 0.000043),
        "gemm_kernel": (207.777999, 0.000820),
    }
    events = {k: (s * 1_000_000.0, d * 1_000_000.0) for k, (s, d) in raw.items()}
    draw_timeline(events, "VTune/ITT (xpu-offload) timeline — randn + matmul",
                  f"{OUT_DIR}/vtune_timeline.png")


if __name__ == "__main__":
    render_iprof()
    render_kineto()
    render_vtune()
