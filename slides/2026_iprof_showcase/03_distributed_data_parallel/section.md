# A real DistributedDataParallel training loop, two nodes

Every prior example traces a single operator or a hand-rolled MPI
collective. This one traces an actual `torch.nn.parallel.DistributedDataParallel`
(DDP) training loop, two Aurora nodes, two ranks, XCCL backend.

The workload is a small, synthetic, Llama-inspired transformer ("TinyLlama"),
adapted from Nathan S. Nichols' THAPI/iprof AI/ML tracing exercise
([`intro-to-tracing-ai-ml-models-with-thapi-slides`](https://github.com/nscottnichols/intro-to-tracing-ai-ml-models-with-thapi-slides),
MIT licensed). It downloads no weights and no dataset; everything is
randomly initialized and tokens are synthetic. Two entry points share one
implementation:

- `train_llama3_demo.py`: no application-level ITT markers.
- `train_llama3_demo_with_itt.py`: the same model and training loop, with
  named ITT regions (`Step.N`, `Forward`, `Layer.i`, `Attn.i`, `MLP.i`,
  `Backward`, `Optimizer.Step`, plus `Init.MPI`/`Init.ProcessGroup`/
  `Init.Model` and `Metrics.AllReduce`) wrapping each phase.

```python
# train_llama3_demo_with_itt.py (excerpt)
import ittapi
from llama3_demo_common import run

def itt_task(name: str, domain: str):
    return ittapi.task(name, domain=domain)

run(task_factory=itt_task, description=__doc__)
```

```python
# llama3_demo_common.py (excerpt): DDP wrapping and the marker hierarchy
model = TinyLlama(...).to(device)
model = torch.nn.parallel.DistributedDataParallel(
    model, broadcast_buffers=False, device_ids=[local_rank], output_device=local_rank
)

for step in range(args.steps):
    with regions.task(f"Step.{step}"):
        inputs, targets = get_batch()
        with regions.task("Forward"):
            logits = model(inputs)
            loss = F.cross_entropy(...)
        with regions.task("Backward"):
            loss.backward()
        with regions.task("Optimizer.Step"):
            optimizer.step()
    if step % 5 == 0:
        with regions.task("Metrics.AllReduce"):
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
```

```bash
export MASTER_ADDR=<host0>
export PALS_WORLD_SIZE=2
mpiexec --no-transfer -n 2 -ppn 1 -hosts <host0>,<host1> \
  ./ccl_local_wrap.sh \
  iprof --backends pytorch,itt --traced-ranks -1 \
  -l iprof_timeline.pftrace --analysis-output iprof_summary.txt -- \
  python3 train_llama3_demo_with_itt.py --device xpu --steps 2 \
    --batch-size 1 --seq-len 16 --dim 64 --layers 1 --heads 4 --vocab-size 128 --bf16
```

`ccl_local_wrap.sh` (from the same reference repo) reads PALS'
`PALS_LOCAL_RANKID`/`PALS_LOCAL_SIZE` and exports them as `CCL_LOCAL_RANK`/
`CCL_LOCAL_SIZE`, which oneCCL needs for intra-node coordination.
`--traced-ranks -1` traces every rank, not just rank 0.

> [!IMPORTANT]
> Two real issues showed up getting this running on this system:
> - PALS here never sets a world-size environment variable (no `PALS_WORLD_SIZE`, `PMI_SIZE`, or `OMPI_COMM_WORLD_SIZE`); the demo's own rank-detection code silently falls back to `world_size=1` on both ranks, so `init_process_group` hangs forever waiting for a peer that never checks in. Exporting `PALS_WORLD_SIZE=2` fixes it. Do not use `PMI_SIZE` instead: MPICH's own PMI auto-detection aborts if it sees both `PMI_SIZE` and `PMIX_NAMESPACE` set at once.
> - Tracing with `--backends ze` (or the default backend set, which includes `ze`) alongside a real two-node XCCL run segfaults the traced process outright. Restricting to `--backends pytorch,itt` avoids the ZE backend's LD_PRELOAD and runs cleanly. Same class of backend-interaction issue as the CXI sampling case, not a DDP/ITT bug.

The tally shows "2 Hostnames" and real DDP collective calls alongside the
ITT markers:

```text
BACKEND_PYTORCH | 2 Hostnames | 2 Processes | 4 Threads |

             Name |    Time | Time(%) | Calls |   Average |      Min |      Max |
 c10d::allgather_ |   1.84s |  31.39% |     2 |  921.95ms | 917.59ms | 926.32ms |
 c10d::broadcast_ | 243.34ms |  4.14% |     8 |   30.42ms |  31.01us |  72.60ms |
 c10d::allreduce_ | 158.15ms |  2.69% |     6 |   26.36ms | 322.67us |  80.04ms |
...

BACKEND_ITT | 2 Hostnames | 2 Processes | 4 Threads |

                                       Name |    Time | Time(%) | Calls |  Average |     Min |     Max |
                      LlamaDemo:Init.Model |   3.40s |  43.58% |     2 |    1.70s |   1.70s |   1.70s |
                          LlamaDemo:Step.0 |   1.31s |  16.81% |     2 | 655.86ms | 652.13ms | 659.58ms |
                         LlamaDemo:Forward | 844.70ms | 10.82% |     4 | 211.17ms |   3.25ms | 421.75ms |
                         LlamaDemo:Layer.0 | 493.21ms |  6.32% |     4 | 123.30ms |   1.87ms | 249.42ms |
                          LlamaDemo:Attn.0 | 475.19ms |  6.09% |     4 | 118.80ms |   1.42ms | 240.89ms |
oneCCL::API:allreduce_scaleout_sycl_simple | 379.52ms |  4.86% |     6 |  63.25ms |  55.32ms |  75.61ms |
                        LlamaDemo:Backward | 334.10ms |  4.28% |     4 |  83.53ms |   4.36ms | 165.31ms |
               LlamaDemo:Init.ProcessGroup | 209.35ms |  2.68% |     2 | 104.68ms |   3.76ms | 205.60ms |
               LlamaDemo:Metrics.AllReduce | 153.30ms |  1.96% |     2 |  76.65ms |  72.94ms |  80.36ms |
                  LlamaDemo:Optimizer.Step | 125.01ms |  1.60% |     4 |  31.25ms | 737.87us |  65.46ms |
...
```

`oneCCL::API:allreduce_scaleout_sycl_simple` is the actual collective kernel
underneath `c10d::allreduce_`/`LlamaDemo:Metrics.AllReduce`; the ITT markers
label the model's own phases, the PyTorch backend shows the dispatcher-level
collective calls.

Both ranks enter `Step.0` almost simultaneously, 29 microseconds apart in
this run, confirming the two processes are in lockstep as DDP requires:

```text
17:24:06.838092847 - x4300c2s6b0n0 - vpid: 678524, vtid: 678524 - lttng_ust_itt:__itt_task_begin: {
  domain: 0x000055e92fc9a140,
  name: 0x000055e930442df0,
  domain__nameA_val: "LlamaDemo",
  name__strA_val: "Step.0"
}
17:24:06.838121379 - x4607c5s7b0n0 - vpid: 153956, vtid: 153956 - lttng_ust_itt:__itt_task_begin: {
  domain: 0x00005624143a4140,
  name: 0x0000562414babc80,
  domain__nameA_val: "LlamaDemo",
  name__strA_val: "Step.0"
}
```

The interval view confirms it with real durations, one row per host:

```text
lttng:host: { hostname = "x4300c2s6b0n0", vpid = 678524, vtid = 678524, ts = 1791480246838092847, backend = 9 }, { name = "LlamaDemo:Step.0", dur = 650853253, err = 0 }
lttng:host: { hostname = "x4607c5s7b0n0", vpid = 153956, vtid = 153956, ts = 1791480246838121379, backend = 9 }, { name = "LlamaDemo:Step.0", dur = 649455311, err = 0 }
```

No application markers appear at all for the baseline entry point; the same
before/after contrast as the ITT Backend section, now at the scale of a
real multi-node training step instead of a single op.

📄 [Full tally](03_distributed_data_parallel/iprof_summary.txt) ·
[Raw trace](03_distributed_data_parallel/raw_trace.txt) ·
[Intervals](03_distributed_data_parallel/intervals.txt) ·
[Aggregations](03_distributed_data_parallel/aggregations.txt)

🔗 [Explore this trace in Perfetto](https://ui.perfetto.dev/#!/?url=https://raw.githubusercontent.com/DonAurelio/notes/main/slides/2026_iprof_showcase/03_distributed_data_parallel/iprof_timeline.pftrace)

> [!NOTE]
> Adapted from [`iprof_demo_llama3`](https://github.com/nscottnichols/intro-to-tracing-ai-ml-models-with-thapi-slides/tree/main/iprof_demo_llama3) by Nathan S. Nichols, MIT License (see [`UPSTREAM_LICENSE.txt`](03_distributed_data_parallel/UPSTREAM_LICENSE.txt)). `llama3_demo_common.py`, `train_llama3_demo.py`, `train_llama3_demo_with_itt.py`, and `ccl_local_wrap.sh` are used unmodified; only the launch command and environment variables were adapted for this system.
