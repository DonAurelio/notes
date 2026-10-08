<a href="">
<img src="argonne-logo.webp"
     align="right"
     width="30%"
     alt="Argonne National Laboratory">
</a>

# THAPI/iprof: Tracing Heterogeneous APIs

1. Introduction
2. Features Showcase: No Code Changes, Open Trace Format, Perfetto Native Timeline
3. Programming-Model-Based Tracing
4. Performance Counter Sampling (CXI)
5. Intel ITT Backend
6. Heterogeneous Programming Models: CPU + XPU
7. Heterogeneous Programming Models: CPU + GPU
8. Distributed Computing (MPI)
9. DistributedDataParallel

**References:** [THAPI repository](https://github.com/argonne-lcf/THAPI) ·
[iprof documentation (ALCF Aurora)](https://docs.alcf.anl.gov/aurora/performance-tools/iprof/)

---

### 1. Introduction

THAPI is a generic framework for heterogeneous applications — one tracer,
not one per programming model, that works across the diversity of languages,
programming models, and domain-specific frameworks HPC applications
actually mix:

| Languages | Prospective Languages | Programming Models | Domain-Based Programming Models |
|---|---|---|---|
| <ul><li>FORTRAN</li><li>C</li><li>C++</li><li>**Python**</li></ul> | <ul><li>Julia</li><li>Lua</li><li>PGAS approaches</li></ul> | <ul><li>**MPI**</li><li>OpenMP</li><li>**CUDA**, **L0**, **ROCm**, **HIP**, **OpenCL**</li><li>SYCL, Kokkos, Raja</li></ul> | <ul><li>Linear algebra: BLAS/LAPACK</li><li>FFTs: cuFFT, FFTWx, MKL FFT</li><li>Low-level AI: cuDNN, clDNN, Intel DNNL</li><li>AI/ML: TensorFlow, Caffe, **PyTorch**</li></ul> |

This plethora of alternatives is entwined, especially since heterogeneous
computing is now the norm — the diagram below shows how just the
lower-level programming models alone can be layered on top of one another:

**Diagram 1 — Programming-model interrelation**

```mermaid
flowchart TD
    subgraph SYCL_group["SYCL"]
        direction TD
        SYCL --> HIP1[HIP]
        SYCL --> OpenCL1[OpenCL]
        SYCL --> L01[L0]
    end

    subgraph OpenMP_group["OpenMP"]
        direction TD
        OpenMP --> OpenCL2[OpenCL]
        OpenMP --> CUDA1[CUDA]
        OpenMP --> L02[L0]
    end

    subgraph OpenCL_group["OpenCL"]
        direction TD
        OpenCL --> L03[L0]
        OpenCL --> CUDA2[CUDA]
    end

    subgraph HIP_group["HIP"]
        direction TD
        HIP --> CUDA3[CUDA]
        HIP --> OpenCL3[OpenCL]
        HIP --> ROCm1[ROCm]
        HIP --> L04[L0]
    end

    subgraph Kokkos_group["Kokkos"]
        direction TD
        Kokkos --> OpenMP2[OpenMP]
        Kokkos --> CUDA4[CUDA]
        Kokkos --> SYCL2[SYCL]
    end
```

PyTorch itself is a microcosm of the same problem: a single Python call
fans out, through ATen's dispatcher, to a different vendor-specific
programming model depending on the device — which is exactly the kind of
heterogeneity a tracer has to handle.

**Diagram 2 — The PyTorch case**

```mermaid
flowchart TD
    subgraph App["Application Layer"]
        Python["Python"]
    end

    subgraph Frontend["PyTorch Frontend"]
        TorchAPI["torch (Python API, autograd)"]
    end

    subgraph ATenLayer["ATen (PyTorch's Operator Layer)"]
        ATen["aten:: ops"]
        Dispatcher["Dispatcher (DispatchKey routing)"]
    end

    subgraph Backends["Device Backends"]
        CUDA["CUDA backend<br>(DispatchKey: CUDA)"]
        ROCm["ROCm/HIP backend<br>(HIPified, same DispatchKey: CUDA)"]
        XPU["XPU backend<br>(DispatchKey: XPU)"]
        CPU["CPU backend<br>(DispatchKey: CPU)"]
        MPS["MPS backend<br>(DispatchKey: MPS)"]
    end

    subgraph Libs["Vendor Libraries / Runtime"]
        CUDALibs["cuBLAS, cuDNN, NCCL<br>CUDA driver"]
        ROCmLibs["rocBLAS, MIOpen, RCCL<br>HIP/ROCr runtime"]
        XPULibs["oneMKL, oneDNN<br>SYCL / Level Zero runtime"]
        CPULibs["MKL, oneDNN"]
        MPSLibs["Metal Performance Shaders"]
    end

    Python --> TorchAPI
    TorchAPI --> ATen
    ATen --> Dispatcher
    Dispatcher --> CUDA
    Dispatcher --> ROCm
    Dispatcher --> XPU
    Dispatcher --> CPU
    Dispatcher --> MPS

    CUDA --> CUDALibs
    ROCm --> ROCmLibs
    XPU --> XPULibs
    CPU --> CPULibs
    MPS --> MPSLibs
```

**Supported backends**

| Backend | Description |
|---|---|
| **MPI** | Message Passing Interface — traces point-to-point and collective calls across distributed-memory processes |
| **OpenMP** | Shared-memory parallelism on the host — traces runtime calls (parallel regions, tasks, synchronization) |
| **OpenCL** | Cross-vendor heterogeneous-compute API — traces host-side `cl*` calls (contexts, command queues, kernels) |
| **Level Zero (L0)** | Intel's low-level GPU programming interface — traces host-side `ze*` calls, the backbone of oneAPI/SYCL on Intel GPUs |
| **CUDA** | NVIDIA's GPU programming model — traces both the CUDA runtime and driver APIs |
| **HIP** | AMD's CUDA-portable GPU programming model — traces `hip*` runtime calls on AMD (and, via HIP's own portability, NVIDIA) GPUs |
| **CXI** | HPE Slingshot's low-level network interface — traces host-side fabric/NIC calls, relevant to performance-counter-style sampling (Section 4) |
| **ITT** | Intel's Instrumentation and Tracing Technology — captures named ranges/tasks apps or libraries (e.g. oneDNN) emit for tools like VTune (Section 5) |
| **PyTorch** | Traces `aten::*` ops directly via PyTorch's own RecordFunction hook — no `torch.profiler`/`emit_itt()` wrapping required |

> [!IMPORTANT]
> **Key takeaway** — THAPI/iprof is not a PyTorch-specific or
> vendor-specific tool: it's one tracer whose backends span the whole stack
> this table and both diagrams describe, from the network fabric (CXI) and
> device driver level (Level Zero, CUDA, HIP) up to a specific AI/ML
> framework's own operator dispatch (PyTorch). The rest of this document
> walks through what that buys you in practice, one feature at a time.
