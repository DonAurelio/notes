# Outline

## 1. How PyTorch Dispatches Ops
- 1.1 The ATen dispatcher — block diagram (image only)
- 1.2 The RecordFunction hook — tracer.cpp extract + trace extract + summary of calls

## 2. Granularity Toggles
- 2.1 The six knobs — bullets (left) / code or toggle table (right)
  *(may split into 2 slides if it's too dense — e.g. scope+thread | depth+inputs+sampling — decide when we get there)*

## 3. Concurrency (CPU threads) — Step 3
- 3.1 Intro — Hogwild vs. intra-op app diagram + variations table
- 3.2 Result — global vs. thread-local under inter-op concurrency (91% missed)
- 3.3 Result — depth counter stays correctly per-thread
- 3.4 Result — intra-op parallelism is invisible to the hook

## 4. GPU Device — Step 4
- 4.1 Intro — same apps on XPU + variations table (DEVICE/SYNC)
- 4.2 Result — autograd moves to its own thread (CPU tie breaks)
- 4.3 Result — multi-thread + device: thread-local misses 91.7%
- 4.4 Result — host-launch vs. device-exec: sync is invisible to the hook
- 4.5 Result — device shows up in the args for free

## 5. MPI, CPU (DDP) — Step 5
- 5.1 Intro — DDP app diagram + variations table
- 5.2 Result — N ranks → N coherent vpids for free
- 5.3 Result — every rank balanced, no cross-rank bleed
- 5.4 Result — on CPU, thread-local ties again (100%)
- 5.5 Result — DDP collectives are ordinary dispatched ops

## 6. MPI, GPU (DDP on XPU) — Step 6
- 6.1 Intro — DDP-on-XPU app diagram + variations table
- 6.2 Result — process coherence holds on device
- 6.3 Result — CPU tie breaks per rank (autograd thread per rank)
- 6.4 Result — thread-local still 100%: it's about the spawner, not the device
- 6.5 Result — collectives + device placement captured for free

That's ~23 content slides + title.
