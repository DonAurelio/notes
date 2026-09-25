# Step — Validating THAPI's own `pytorch` backend (XPU, MPI, MPI+XPU)

[02_pytorch_tracing](../02_pytorch_tracing/) built a **standalone** RecordFunction
tracer from scratch (`LD_PRELOAD` + hand-written LTTng-UST provider) to understand
the PyTorch dispatcher hook mechanism. This step switches to
[THAPI](https://github.com/argonne-lcf/THAPI) (Tracing Heterogeneous APIs) itself
— specifically its built-in `pytorch` backend, driven through the `iprof` CLI —
and validates it across three axes 02_pytorch_tracing already explored with the
custom tracer: **device** (CPU vs XPU), **multi-process** (MPI ranks), and
**multi-node** (ranks spread across physical nodes). The tracer under test is no
longer ours — it's THAPI's shipped `libTracerPytorch.so` + `iprof` orchestration —
so the question is whether *THAPI's own* pipeline holds up under the same
conditions, not whether RecordFunction itself works (already established in Step
02).

The workload is the same idea as 02_pytorch_tracing's `model.py`: a tiny
`Linear(10,5) -> relu -> sum -> backward`, deterministic (`torch.manual_seed(0)`),
toggled between CPU and XPU by a `DEVICE` env var. THAPI was already built +
installed from a separate clone (branch `pytorch-backend-analysis`); building
THAPI itself is out of scope here — see `thapi-tracer-environment-config` /
`thapi-tracer-build` if you need to reproduce that.

| Toggle | Values | Effect |
|---|---|---|
| `DEVICE` | `cpu` (default) \| `xpu` | run the model on CPU or an Aurora XPU tile |
| ranks | `1` \| `2` | run standalone or under `mpirun -n 2` |
| nodes | `1` \| `2` | ranks co-located on one compute node, or one rank per node |

`DEVICE` carries over from 02_pytorch_tracing; ranks/nodes are the new axes here,
combined into 6 experiments (00–05, see contents below). XPU and MPI experiments
require a real Aurora compute node — the login node has 0 XPUs and cannot even
initialize MPI (`mpirun -n 2 hostname` exits 127 there).

## Contents

```
03_thapi_pytorch_tracing/
├── README.md                       # this file
├── env.sh                          # module recipe (oneapi -> ruby-* -> gmake/protobuf -> babeltrace2/lttng-tools -> [mpich] -> frameworks LAST)
├── example/
│   └── model.py                    # Linear(10,5) -> relu -> sum -> backward; DEVICE-toggled, PALS_LOCAL_RANKID-aware
├── analyze_thapi.py                # parses THAPI's raw-trace text into entry/exit balance, vtid/rank counts, backward-thread detection
├── 00_cpu_baseline/
│   └── traces/                     # raw.txt (1722 lines), interval.txt (861), tally.txt (67), timeline.pftrace
├── 01_xpu_single/
│   └── traces/                     # raw.txt (1838), interval.txt (919), tally.txt (74), timeline.pftrace
├── 02_mpi_cpu/
│   └── traces/                     # raw.txt (3444), interval.txt (1722), tally.txt (67), timeline.pftrace
├── 03_mpi_xpu/
│   └── traces/                     # raw.txt (3676), interval.txt (1838), tally.txt (74), timeline.pftrace
├── 04_mpi_2node_cpu/
│   └── traces/                     # raw.txt (3444), interval.txt (1722), tally.txt (67), timeline.pftrace
└── 05_mpi_2node_xpu/
    └── traces/                     # raw.txt (3676), interval.txt (1838), tally.txt (74), timeline.pftrace
```

Each experiment's `traces/` holds THAPI's three text views of the same recorded
session plus the Perfetto timeline:

- `raw.txt` — `babeltrace_thapi trace`: one line per `op_entry`/`op_exit` event.
- `interval.txt` — `babeltrace_thapi to_interval` + `babeltrace2`: one line per
  completed call (start + `dur`), THAPI's merged entry/exit representation.
- `tally.txt` — `iprof`'s default aggregated Name/Time/Time%/Calls/Average/Min/Max
  table.
- `timeline.pftrace` — `iprof -l`, openable in [Perfetto](https://ui.perfetto.dev).

## How to reproduce

```bash
THAPI_INSTALL=/path/to/THAPI/install source env.sh [mpi]   # mpi arg needed for 02-05
cd example

# 00 — login node, CPU, single process
DEVICE=cpu iprof --backends pytorch -t --trace-output <dir> -- python3 model.py

# 01 — compute node, XPU, single process
DEVICE=xpu iprof --backends pytorch -t --trace-output <dir> -- python3 model.py

# 02 — compute node, MPI, 2 ranks co-located, CPU
DEVICE=cpu mpirun -n 2 -- iprof --backends mpi,pytorch -t --trace-output <dir> -- python3 model.py

# 03 — compute node, MPI, 2 ranks co-located, XPU (rank r -> xpu:r via PALS_LOCAL_RANKID)
DEVICE=xpu mpirun -n 2 -- iprof --backends mpi,pytorch -t --trace-output <dir> -- python3 model.py

# 04 — 2 compute nodes, 1 rank/node, CPU (trace-output MUST be under $HOME, not node-local /tmp)
DEVICE=cpu mpirun --hosts <node1>,<node2> -ppn 1 -n 2 -- iprof --backends mpi,pytorch -t \
  --trace-output <shared_dir> -- python3 model.py

# 05 — 2 compute nodes, 1 rank/node, XPU (each rank alone on its node -> xpu:0 on both)
DEVICE=xpu mpirun --hosts <node1>,<node2> -ppn 1 -n 2 -- iprof --backends mpi,pytorch -t \
  --trace-output <shared_dir> -- python3 model.py

# from the same --trace-output dir, render the other views
# (pass every <hostname> subdir as a separate positional arg for multi-node 04/05)
babeltrace_thapi trace --context --backends pytorch:6 -- <dir>/<hostname> [<dir>/<hostname2> ...] > raw.txt
babeltrace_thapi to_interval --output <interval_dir> -- <dir>/<hostname> [<dir>/<hostname2> ...]
babeltrace2 --plugin-path "$THAPI_INSTALL/lib" <interval_dir>                                     > interval.txt
iprof --backends pytorch[,mpi] --analysis-output tally.txt -r <dir>                                # tally
iprof --backends pytorch[,mpi] -l timeline.pftrace -r <dir>                                        # timeline
```

**`iprof` wraps each MPI rank — it does not wrap `mpirun`.** `mpirun -n 2 -- iprof
...` is correct; `iprof -- mpirun ...` is explicitly the wrong direction (`iprof`
itself warns against it) and only profiles a single hostname.

`analyze_thapi.py <raw.txt>` reproduces the per-experiment entry/exit-balance,
vtid/rank, and backward-thread numbers cited below — THAPI's own raw-trace line
schema (`vtid: N`, unquoted `name: ...`) differs from 02_pytorch_tracing's
hand-rolled tracer schema, so a schema-specific parser was needed rather than
reusing that stage's analysis tooling.

## The results

### THAPI's pytorch backend traces cleanly on CPU — the control run
The single-process CPU run is the baseline every other experiment is compared
against:

```
00_cpu_baseline:  1722 events (861 entry / 861 exit), balanced=True
  1 (host,vpid,vtid) key: 861 entry / 861 exit / 8 backward — inline, same thread
tally: 861 calls, top op aten::linear (33.83%)
```

### XPU adds a second, dedicated autograd-engine thread — same finding as Step 4, in THAPI's own tracer
Moving to `DEVICE=xpu` (still 1 process) makes PyTorch spin up a second thread
purely for backward — the same device effect 02_pytorch_tracing's
[Step 4](../02_pytorch_tracing/04_gpu_device/README.md) found with the
hand-written tracer, now confirmed with THAPI's shipped one:

```
01_xpu_single:  1838 events (919 entry / 919 exit), balanced=True
  2 (host,vpid,vtid) keys:
    main thread:            886 entry / 886 exit / 0 backward
    autograd-engine thread:  33 entry /  33 exit / 8 backward
tally: "2 Threads", top op shifts to aten::masked_select (17.37%) — XPU copy/index overhead
```

### MPI + pytorch backends combined: no existing THAPI test for this, validated here
THAPI's own repo has no bats test or example combining the `mpi` and `pytorch`
backends. Running `mpirun -n 2 -- iprof --backends mpi,pytorch -- python3
model.py` (CPU, 2 co-located ranks) shows each rank tracing exactly as if it were
alone:

```
02_mpi_cpu:  3444 events (1722/1722), balanced=True — exactly 2x the CPU baseline
  2 (host,vpid,vtid) keys, one per rank: 861/861/8 each — byte-identical to 00
tally: "2 Processes | 2 Threads", 1722 calls
```

The XPU + MPI combination (`03_mpi_xpu`) shows the same per-rank structure as
`01_xpu_single`, doubled, with each rank's device autograd thread staying
correctly attributed to its own rank:

```
03_mpi_xpu:  3676 events (1838/1838), balanced=True — exactly 2x 01_xpu_single
  4 (host,vpid,vtid) keys, 2 per rank: 886/886/0 (main) + 33/33/8 (autograd), per rank
tally: "2 Processes | 4 Threads", 1838 calls
```

### Multi-node MPI: THAPI aggregates correctly across physical nodes, not just across ranks
Experiments 02/03 could only exercise same-node rank aggregation (both ranks
share one hostname). Spreading the same 2 ranks across 2 real compute nodes
(`mpirun --hosts n1,n2 -ppn 1 -n 2`) shows THAPI's master-election correctly
merging per-node output — the tally now reports **2 Hostnames**, and every
per-rank number matches the same-node experiments exactly:

```
04_mpi_2node_cpu:  3444 events, balanced=True — identical total to 02_mpi_cpu
  2 hostnames, 1 key each: 861/861/8 per rank — same as 02, now on 2 nodes
tally: "2 Hostnames | 2 Processes | 2 Threads"

05_mpi_2node_xpu:  3676 events, balanced=True — identical total to 03_mpi_xpu
  2 hostnames, 2 keys each: 886/886/0 (main) + 33/33/8 (autograd), per rank per node
tally: "2 Hostnames | 2 Processes | 4 Threads"
```

`babeltrace_thapi trace`/`to_interval` accept multiple hostname-subdir arguments
directly, so no manual trace-merging step was needed to combine the two nodes'
data; pointing `--trace-output` at the shared `$HOME` filesystem (rather than a
node-local path) was enough for both nodes' `iprof` instances to land in one
accessible tree.

## What this demonstrates

- **THAPI's shipped `pytorch` backend reproduces the device finding from
  02_pytorch_tracing's hand-written tracer** — backward moves to a dedicated
  autograd-engine thread on XPU, inline on CPU — without needing our own `.so`.
- **`mpi`+`pytorch` backends combine correctly, same-node and cross-node**, a
  combination with no prior test coverage in THAPI itself: per-rank event counts
  are byte-identical to the single-process runs, with zero cross-rank or
  cross-node contamination in either the numeric results or the trace's
  `(host,vpid,vtid)` attribution.
- **Node topology is transparent to trace correctness.** The 2-node experiments
  (04/05) produce exactly the same per-rank numbers as their same-node
  counterparts (02/03) — only the hostname count in the tally changes.
- **`iprof` must wrap each rank, never wrap `mpirun`** — confirmed both from
  THAPI's own runtime warning and from every experiment here using the correct
  form.
- **All three of THAPI's own trace views agree.** Raw event count, interval-line
  count, and tally total-calls count match exactly in all six experiments — no
  discrepancy anywhere in this validation.
