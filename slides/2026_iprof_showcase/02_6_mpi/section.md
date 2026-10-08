#### 2.6 Distributed Computing (MPI)

`mpirun iprof -- <executable>` wraps every rank, on every node, in its own
THAPI capture; the tally then merges them into one view across hosts. The
base code gets one addition: an MPI collective, so there's real MPI
traffic to trace alongside PyTorch and the device backends.

```bash
mpirun -n 2 -ppn 1 -hosts <host0>,<host1> iprof --analysis-output iprof_summary.txt -- python model.py
```

```python
# model.py
import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

x = torch.randn(2048, 2048, device="xpu")
y = torch.matmul(x, x)
torch.xpu.synchronize()

local_sum = y.sum().item()
total_sum = comm.allreduce(local_sum, op=MPI.SUM)
print(f"rank {rank}: local_sum={local_sum:.4f} total_sum={total_sum:.4f}")
```

Two ranks, two nodes, one tally. Every backend section now reports "2
Hostnames", and a new `BACKEND_MPI` section appears for the
`comm.allreduce` call:

```text
BACKEND_PYTORCH | 2 Hostnames | 2 Processes | 2 Threads |

         Name |     Time | Time(%) | Calls |  Average |     Min |     Max |
    aten::sum | 157.38ms |  26.46% |     2 |  78.69ms | 75.98ms | 81.40ms |
 aten::matmul | 133.51ms |  22.44% |     2 |  66.76ms | 66.66ms | 66.85ms |
...

BACKEND_MPI | 2 Hostnames | 2 Processes | 2 Threads |

            Name |   Time | Time(%) | Calls |  Average |      Min |      Max |
 MPI_Init_thread |  1.36s |  97.89% |     2 | 678.08ms | 591.83ms | 764.32ms |
    MPI_Finalize | 23.87ms |   1.72% |     2 |  11.94ms |  11.86ms |  12.01ms |
    MPI_Comm_dup |  4.90ms |   0.35% |     2 |   2.45ms | 101.30us |   4.80ms |
      MPI_Bcast_c | 101.42us | 0.01% |     4 |  25.35us |   4.00us |  40.96us |
...

BACKEND_OPENCL,BACKEND_ZE | 2 Hostnames | 2 Processes | 2 Threads |
...
```

`allreduce` isn't one call on the wire; `mpi4py` implements it as a
`Comm_dup` followed by point-to-point `Send`/`Recv` and a `Bcast` back
out. The raw trace shows both ranks' MPI calls interleaved by real wall
clock time, one line per host:

```bash
mpirun -n 2 -ppn 1 -hosts <host0>,<host1> iprof --trace -- python model.py
```

```text
16:25:28.363166942 - x4420c3s0b0n0 - vpid: 557966, vtid: 557966 - lttng_ust_mpi:MPI_Comm_dup_entry: { comm: 0x0000000044000000, newcomm: 0x000055d5ccf02810 }
16:25:28.366546713 - x4420c3s1b0n0 - vpid: 544215, vtid: 544215 - lttng_ust_mpi:MPI_Comm_dup_entry: { comm: 0x0000000044000000, newcomm: 0x0000557d2dcc9790 }
16:25:28.366626971 - x4420c3s0b0n0 - vpid: 557966, vtid: 557966 - lttng_ust_mpi:MPI_Recv_c_entry: { buf: 0x00007fffba86db40, count: 1, datatype: 0x000000004c000845, source: 1, tag: 0, comm: 0x0000000084000000, status: 0x0000000000000001 }
16:25:28.366668133 - x4420c3s1b0n0 - vpid: 544215, vtid: 544215 - lttng_ust_mpi:MPI_Send_c_entry: { buf: 0x00007ffdd16eb490, count: 1, datatype: 0x000000004c000845, dest: 0, tag: 0, comm: 0x0000000084000000 }
16:25:28.366684705 - x4420c3s0b0n0 - vpid: 557966, vtid: 557966 - lttng_ust_mpi:MPI_Bcast_c_entry: { buffer: 0x00007fffba86dbc0, count: 1, datatype: 0x000000004c000845, root: 0, comm: 0x0000000084000000 }
16:25:28.366701166 - x4420c3s1b0n0 - vpid: 544215, vtid: 544215 - lttng_ust_mpi:MPI_Bcast_c_entry: { buffer: 0x00007ffdd16eb510, count: 1, datatype: 0x000000004c000845, root: 0, comm: 0x0000000084000000 }
```

One `hostname` field per event is what makes the two-node merge possible;
nothing in the base code changes between a single-node and a multi-node
run; only the `mpirun -hosts` list does.

📄 [Full tally](02_6_mpi/iprof_summary.txt) ·
[Raw trace](02_6_mpi/raw_trace.txt) ·
[Intervals](02_6_mpi/intervals.txt) ·
[Aggregations](02_6_mpi/aggregations.txt)

🔗 [Explore this trace in Perfetto](https://ui.perfetto.dev/#!/?url=https://raw.githubusercontent.com/DonAurelio/notes/main/slides/2026_iprof_showcase/02_6_mpi/iprof_timeline.pftrace)
