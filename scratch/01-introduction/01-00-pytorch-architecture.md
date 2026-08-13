# PyTorch architecture

![PyTorch architecture ](../img/01-introduction/pytorch_architecture_with_setprofile.svg)

### Layers

1. **User code / nn.Module** — groups tensor operations into reusable model building blocks.
2. **torch.nn / functional** — what users interact with; the Python API where the user calls tensor operations (add, matmul, softmax…)
3. **[Autograd engine](./01.1-autograd-engine.md)** — records each operation into a graph as it runs, later used to compute gradients
4. **[Dispatcher (ATen)](./01.2-dispatcher.md)** ⟵ *our hook* — for every operation, picks the right implementation based on device (CPU/GPU) and data type
5. **C10 core** — defines the basic building blocks: Tensor, Storage, Device
6. **Backend kernels** — the device-specific code that does the actual math

### Hook points




# References

1. [The PyTorch architecture stack](https://developers.redhat.com/articles/2026/02/19/understanding-aten-pytorchs-tensor-library#what_is_aten_)


