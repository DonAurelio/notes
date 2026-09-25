# THAPI PyTorch tracing

This folder explores a simple question: **can we watch what PyTorch does under the
hood without changing a single line of the Python program?**

The answer is yes. PyTorch has a built-in hook called **RecordFunction** that fires
on every operation it runs. We attach a small tracer to it from the outside, let a
normal PyTorch script run, and record everything it did to a trace file we can read
back later.

## What's inside

| Folder | What it's about |
|---|---|
| `00_pytorch_introduction/` | Beginner notes on PyTorch and machine learning — start here. |
| `01_lttng_recordfunction_trace/` | The first working trace of an unmodified PyTorch program. |
| `02_recordfunction_granularities/` | The different levels of detail we can record. |
| `03_concurrency_device/` | What changes when the program uses many CPU threads. |
| `04_gpu_device/` | What changes when the program runs on a GPU. |
| `05_mpi_cpu/` | What changes when the program runs as many processes at once (on CPU). |
| `06_mpi_gpu/` | The same many-process program, this time on the GPU. |

Each folder has its own README that explains the idea, how to reproduce it, and
what we found. Read them in order.

## What we learned

- We can trace an unmodified PyTorch program from the outside — no code changes.
- The trace records each operation once, at the level a user wrote it, not every internal step.
- How much detail we capture is a choice we can turn up or down.
- To catch everything, the tracer must watch all threads, not just the main one.
- On a GPU, PyTorch runs part of the work on its own thread, so watching all threads becomes essential.
- The trace shows when work is *launched*, not how long it takes on the GPU.
- When the program runs as many processes at once, each one is traced cleanly on its own, and the messages they send each other to stay in sync show up too.
- Running those many processes on GPUs instead of CPUs traces just as cleanly, and each one shows which GPU it used.
