# Dispatcher (ATen)

PyTorch's ATen dispatcher is the mechanism that determines how an operation, such as `torch.add`, should be handled at runtime. 

An operator  (e.g., `torch.add`) represents a higher-level computation. The actual work performed for that operation can involve multiple *dispatch layers*, depending on the *active dispatch keys*.
The number of dispatch layers and keys depend on the properties of the tensors involved in the operation and the PyTorch execution context.

```text
torch.add(a, b)
   │
   │  Active dispatch keys
   │  {Autograd, Autocast, CUDA}
   ▼
ATen Dispatcher
   │
   │  Active dispatch keys
   │  {Autograd, Autocast, CUDA} (Highest-priority key first)
   ▼
┌──────────────────────────────────────┐
│ Autograd dispatch key                │
│                                      │
│ → Autograd implementation            │
│ → Handle gradient tracking           │
└──────────────────┬───────────────────┘
                   │
                   │ continues dispatch
                   ▼
┌──────────────────────────────────────┐
│ Autocast dispatch key                │
│                                      │
│ → Autocast implementation            │
│ → Handle mixed-precision behavior    │
└──────────────────┬───────────────────┘
                   │
                   │ continues dispatch
                   ▼
┌──────────────────────────────────────┐
│ CUDA dispatch key                    │
│                                      │
│ → CUDA implementation / kernel       │
│ → Perform the addition on the GPU    │
└──────────────────┬───────────────────┘
                   │
                   ▼
                 Result
```

We would distinguish three concepts:

* **Operator** → what computation is being requested (`add`)
* **Dispatch key** → what aspect/context of the computation needs special handling (Autograd, CUDA, Autocast, etc.)
* **Implementation/kernel** → the code that handles that particular aspect (i.e., *dispatch key*).

One subtle point: a dispatch key is not itself an implementation. It is an identifier used by the dispatcher to select an implementation.

The final computation may involve more than one implementation as the operation moves through the dispatch system.

## The core problem it solves

A single operator like `add` needs different implementations depending on: the device (CPU, CUDA, MPS, XLA...), the dtype/autograd requirements (does it need gradient tracking?), whether it's a sparse or dense tensor, whether autocast/mixed precision is active, whether you're inside a `vmap` or tracing context, etc. Rather than hardcoding a giant if/else chain, PyTorch factors each of these concerns into a separate **dispatch key**, and the dispatcher composes them.

Suppose you write:

```python
c = torch.add(a, b)
```

You are simply saying:

> "**Add** a **and** b."

But PyTorch needs to consider several properties of the inputs and the current execution context to determine how an operation should be handled at runtime. Some of these properties are represented by dispatch keys, while others influence dispatch through other mechanisms. Examples of dispatch keys include `CPU`, `CUDA`, `Autograd`, `AutocastCUDA`, `Tracer`, `Conjugate`, `Functionalize`, `Python` (for `__torch_dispatch__` subclasses), and `Batched` (for `vmap`).


| Question                    | Per tensor or whole operation? | Example                                               |
| --------------------------- | ------------------------------ | ----------------------------------------------------- |
| CPU or CUDA?                | **Per tensor**                 | `a → CUDA`, `b → CUDA` → `CUDA` key                   |
| Dense or sparse?            | **Per tensor**                 | `a → Dense`, `b → Dense` → `Strided` key              |
| What dtype?                 | **Per tensor**                 | `a → float32`, `b → float32`                          |
| Do we need autograd?        | **Operation/context**          | Gradient tracking is active → `Autograd` key          |
| Is mixed precision enabled? | **Operation/context**          | Autocast is active → `AutocastCUDA` key               |
| Are we inside `vmap`?       | **Operation/context**          | The current execution is under `vmap` → `Batched` key |
| Are we tracing?             | **Operation/context**          | The operation is being traced → tracing-related key   |

Without dispatch keys, PyTorch would need one enormous `add` function containing conditions for every possible device, dtype, execution mode, and feature:

```python
def add(a, b):
    if device == CPU:
        if dtype == float32:
            if autograd:
                ...
            else:
                ...
        elif dtype == float64:
            ...

    elif device == CUDA:
        if dtype == float32:
            ...
        elif dtype == float16:
            ...

    # And then handle MPS, sparse tensors,
    # autocast, vmap, tracing, etc.
```

This would quickly become difficult to maintain, because every new device or feature would add more conditions and combinations. **Dispatch keys solve this problem by separating these concerns into different implementations that the dispatcher can select and compose at runtime.**

## Example: adding two CUDA tensors

```python
a = torch.tensor([1, 2, 3], device="cuda", requires_grad=True)
b = torch.tensor([4, 5, 6], device="cuda", requires_grad=True)

c = torch.add(a, b)
```

Conceptually, the flow looks like this

```
             torch.add(a, b)
                    │
                    ▼
             ┌─────────────┐
             │  Dispatcher │
             └──────┬──────┘
                    │
                    │ Gather dispatch keys
                    ▼
             {CUDA, Autograd}
                    │
                    ▼
          Choose highest-priority key
                    │
                    ▼
                Autograd
                    │
                    ▼
       Autograd add implementation
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
       Record operation   Call lower
       for backward       dispatch key
                           │
                           ▼
                          CUDA
                           │
                           ▼
                    CUDA add kernel
                           │
                           ▼
                      GPU computes
                           │
                           ▼
                           c
```

The key idea is:

> Autograd may get the first opportunity to handle the operation. It can do some work related to gradient tracking and then allow the operation to continue to the CUDA implementation.

__Step 1 - Gather the dispatch keys__

The dispatcher collects all the dispatch keys associated with the input tensors and the current execution context, and combines them into one set of keys.

```
Inputs
 ├── a → CUDA
 └── b → CUDA

Execution context
 │   requires_grad=True
 └── Autograd is relevant.

             ↓

       Active keys
       ┌─────────────┐
       │    CUDA     │
       │   Autograd  │
       └─────────────┘
```

__Step 2 - Choose the highest-priority key__

Picks the **highest-priority** key in that set according to a fixed priority ordering.

```
Higher priority
      │
      ▼
   Autograd
      │
     CUDA
      │
     CPU
      │
      ...
      │
Lower priority
```

The actual ordering is more complicated than this simplified example, but the idea is:

> If several keys are active, the dispatcher follows a predefined priority order and selects the highest-priority applicable key.

So imagine the `Autograd` was already handled and the next selected key (layer) is:

```
CUDA
```

The dispatcher now knows:

> "I need the add implementation registered for CUDA."

__Step 3 - Look up the kernel__

Every ATen operator has a dispatch table.

You can imagine the `add` table looking something like this:

```
                 add dispatch table

Dispatch Key          Implementation
────────────────────────────────────────
CPU             →     CPU add kernel
CUDA            →     CUDA add kernel
MPS             →     MPS add kernel
AutogradCPU     →     Autograd implementation
AutogradCUDA    →     Autograd implementation
...
```

The dispatcher essentially performs:

```
             operator = add
                  +
             key = CUDA
                  │
                  ▼
          ┌─────────────────┐
          │  add dispatch   │
          │     table       │
          ├─────────────────┤
          │ CPU    → ...    │
          │ CUDA   →  ███   │ ← found it
          │ MPS    → ...    │
          │ ...             │
          └─────────────────┘
```

It finds the implementation registered for that operator and dispatch key.

__Step 4 - Execute the kernel__

The term "kernel" comes from GPU/HPC terminology. A computational kernel is a function that performs a specific computation over data.

For example, conceptually, a CPU implementation might look like:

```c++
void add_cpu(Tensor a, Tensor b, Tensor output) {
    for (...) {
        output[i] = a[i] + b[i];
    }
}
```

A CUDA implementation might instead launch GPU threads:

```c++
__global__ void add_cuda(...) {
    int i = ...;
    output[i] = a[i] + b[i];
}
```

# Summary 

PyTorch's dispatcher does two distinct jobs — routing the forward call through the layered keys (building the autograd graph along the way), and then, separately, the autograd engine walks that graph backward, invoking each node's backward kernel through the dispatcher again. Here's the forward path first:

```
torch.add(a, b)
      │
      ▼
 Autograd layer (layered key)
      │
      │ Record information needed
      │ for backward computation
      ▼
 CUDA layer (layered key)
      │
      ▼
 CUDA add kernel
      │
      ▼
 GPU performs addition
```

Higher-priority kernels are typically **wrapper kernels**: they do their job (e.g., record the op for autograd, cast dtypes for autocast) and then explicitly **redispatch** — call back into the dispatcher, but with their own key masked out of the key set, so the next-highest key's kernel runs. This continues down through the layers until it hits a real backend kernel that actually computes the result.

For example, calling `add` on a CUDA tensor with `requires_grad=True`:
- Dispatcher sees `{Autograd, CUDA}` in the key set → picks `Autograd` (higher priority).
- The autograd kernel builds the backward graph node, then redispatches with `Autograd` removed from the set.
- Now only `CUDA` remains → the CUDA kernel runs and does the actual computation.

