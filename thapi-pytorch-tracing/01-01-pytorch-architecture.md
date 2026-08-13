# PyTorch: Architecture

<img src="./img/pytorch_architecture_layers.svg" alt="Perceptron" width="600">

___

### User code / torch.nn

* Neural networks can be constructed using the `torch.nn` package.
* While raw PyTorch tensors and autograd provide the mathematical backend, `torch.nn` acts as a high-level abstraction layer that encapsulates data state, learnable weights, and common architectural patterns.
* `torch.nn.Module` is the fundamental base class used to build and organize all neural network models and layers.
* A typical training procedure for a neural network is as follows:
    - Define the neural network that has some learnable parameters (or weights)
    - Iterate over a dataset of inputs
    - Process input through the network
    - Compute the loss (how far is the output from being correct)
    - Propagate gradients back into the network’s parameters
    - Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

___

**Code 1** is a simple feed-forward network (see **Figure 1**). It takes the input, feeds it through several layers (one in this case) one after the other, and then finally gives the output.

**Code 1.** Single-Layer Feedforward Network

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        """ Define the layers """ 
        super().__init__()
        # an affine operation: y = Wx + b
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        """ Connect the layers """ 
        return torch.relu(self.linear(x))

model = MyModel()

print(model)
```

**Figure 1.** Single-Layer Feedforward Network

<img src="./img/single_layer_feedforwad_network.png" alt="Perceptron" width="600">

___





# References

1. [What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html#what-is-torch-nn-really)
2. [Neural Networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
3. [Understanding ATen: PyTorch's tensor library](https://developers.redhat.com/articles/2026/02/19/understanding-aten-pytorchs-tensor-library#)
    - [The PyTorch architecture stack](https://developers.redhat.com/articles/2026/02/19/understanding-aten-pytorchs-tensor-library#the_pytorch_architecture_stack)
    - [What is ATen?](https://developers.redhat.com/articles/2026/02/19/understanding-aten-pytorchs-tensor-library#what_is_aten_)
    -  [The dispatch system](https://developers.redhat.com/articles/2026/02/19/understanding-aten-pytorchs-tensor-library#core_architecture_components)