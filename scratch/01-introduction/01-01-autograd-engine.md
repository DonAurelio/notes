# PyTorch's autograd

PyTorch's autograd is a reverse-mode automatic differentiation system built around a dynamically constructed computation graph. Here's how it actually works under the hood.

## 1. Building the graph (forward pass)
Every torch.Tensor has a requires_grad flag. When you do operations on tensors that require gradients, PyTorch doesn't just compute the result — it also records how that result was produced.

- Each tensor that's the output of an operation gets a .grad_fn attribute pointing to a Node (technically a Function subclass instance, e.g. AddBackward0, MulBackward0) that knows how to compute the local gradient for that operation.
- Each Node stores references to the Nodes that produced its inputs — this forms a directed acyclic graph (DAG), with leaves being the original input tensors (parameters, inputs) and the root being the final output (usually a scalar loss).
- This graph is built dynamically, on the fly, as each operation actually executes — this is PyTorch's "define-by-run" approach, as opposed to TensorFlow 1.x's static graph. That's why you can use normal Python control flow (loops, if statements) and the graph will just reflect whatever actually ran.
- Leaf tensors that require grad are tracked, and PyTorch keeps a reference to enough intermediate data (saved tensors) to compute gradients later — this is why activations sitting in the graph consume memory until backward is called.

## 2. Triggering the backward pass
When you call .backward() on a scalar output (or pass a gradient argument for non-scalar outputs), PyTorch:

- Starts at the root node with an initial gradient (implicitly 1.0 for scalars, dL/dL).
- Walks the graph backward, from outputs to inputs.

Let me visualize this graph structure and traversal:

![PyTorch autograd forward and backward pass](../img/01-introduction/pytorch_autograd_flow.svg) 

The graph in the diagram: x and w feed into a Mul node, that result plus b feeds into an Add node, producing loss. Each op node stores a grad_fn — its own local derivative rule.

For loss.backward():

1. PyTorch seeds the root gradient (1.0 for a scalar loss).
2. It calls each node's backward function in reverse topological order — this is handled by a C++ engine (torch/csrc/autograd/engine.cpp), not pure Python, for speed.
3. Each Node.backward() takes the gradient of the output w.r.t. the loss and, using the chain rule, computes the gradient of its own inputs. For example AddBackward0 just passes the incoming gradient through unchanged to both operands (since d(a+b)/da = d(a+b)/db = 1). MulBackward0 computes grad_x = grad_output * w and grad_w = grad_output * x.
4. When a node has multiple outgoing edges (its value was used in more than one place), gradients are summed/accumulated at that node — this is why .grad accumulates rather than overwrites, and why you need optimizer.zero_grad() between iterations.
5. When the walk reaches a leaf tensor with requires_grad=True, the computed gradient is written into that tensor's .grad attribute (or accumulated there).

## Key implementation details worth knowing

- **Topological ordering & parallelism**: The engine determines a valid execution order via dependency counting (each node knows how many gradients it's waiting on before it can fire), and it can execute independent branches of the graph in parallel across threads.
- **Dynamic graph, freed after use**: By default, the graph (and any saved intermediate tensors needed for backward) is freed after one .backward() call to save memory. Calling .backward() twice without retain_graph=True raises an error, because the graph is already gone.
- **torch.no_grad() / .detach()**: These stop graph-building entirely — operations still run, but no grad_fn is attached, which is why they're used for inference or freezing parameters.
- **Custom ops**: You can define your own graph node by subclassing torch.autograd.Function and implementing forward() and backward() — this is exactly how built-in ops like Mul and Add are implemented internally.
- **Higher-order gradients**: If you pass create_graph=True to .backward(), the backward computation itself is tracked in a new graph, letting you differentiate through the gradient computation (needed for things like Hessian-vector products or MAML-style meta-learning).

The elegance of the design is that autograd doesn't need to know about "loss functions" or "layers" — it only ever needs each primitive operation to supply a local Jacobian-vector product rule, and the chain rule composes them automatically across an arbitrarily deep, dynamically-built graph.

## What is a gradient, how is it computed? and why is it important?

A gradient is the vector of partial derivatives of a function with respect to each of its inputs. If you have a function L(x, y, z), the gradient is:

∇L = (∂L/∂x, ∂L/∂y, ∂L/∂z)

Each component tells you: "if I nudge this one variable slightly, holding everything else fixed, how much does the output change, and in which direction?" Put together, the gradient vector points in the direction of steepest ascent of the function at that point — the direction you'd move to increase L fastest. Its negative points toward steepest descent.

### How it's computed
There are a few ways, each with tradeoffs:

- **Analytically** — work out the derivative formula by hand (e.g. d(x²)/dx = 2x). Exact, but tedious or infeasible for huge composite functions like neural networks.
- **Numerically** — approximate it via finite differences: (f(x+h) − f(x))/h for small h. Simple but slow (needs a forward pass per parameter) and imprecise due to floating-point error.
- **Automatically (autograd)** — what PyTorch does. Break the function into elementary operations, each with a known local derivative, then apply the chain rule to compose them across the whole computation graph, as outlined above. This gives you exact derivatives (up to floating point precision) at the cost of one backward pass, regardless of how many parameters you have.

Concretely, for a scalar loss L and a graph of operations, autograd computes ∂L/∂θ for every parameter θ by walking the graph backward and multiplying local Jacobians together — that's the whole engine described above.

### Why it matters
The gradient is the compass for optimization. In gradient descent, you update parameters by stepping in the negative gradient direction — the direction of steepest decrease of the loss:

θ ← θ − η·∇L(θ)

where η is the learning rate. Let me show the intuition visually:

![Gradient descent on a loss surface](../img/01-introduction/gradient_descent_intuition.svg)

A few reasons this matters so much in practice:

- **It's the only local information you need.** You don't need to know the whole shape of the loss landscape — just the slope right where you're standing — to make progress toward a minimum. That locality is what makes training massive models with billions of parameters tractable.
- **Sign tells you direction, magnitude tells you steepness.** A large gradient component means that parameter has a big effect on the loss right now; a near-zero gradient means you're near a flat region (a minimum, saddle point, or plateau).
- **It generalizes to any differentiable function.** Neural nets, physics simulations, control systems, financial models — anywhere you can define a differentiable objective, gradients tell you how to improve it.
- **It's what makes backpropagation possible.** Backprop is essentially "compute the gradient of the loss with respect to every weight, efficiently, via the chain rule" — that's precisely what autograd's reverse traversal does, as shown in the earlier diagram.

Without gradients, you'd be stuck doing blind or random search over parameter space — computationally hopeless for anything with more than a handful of parameters. Gradients turn optimization into a directed walk instead of a guessing game.

## Is there a separate graph for the forward pass and the backward pass, to compute gradients?

Not quite — there's really only one graph, built once during the forward pass. The backward pass doesn't build a second graph; it traverses the same graph in reverse, calling each node's stored backward function as it goes.

Here's the distinction that matters:

- **Forward pass**: as each operation runs, PyTorch creates a Node (like `MulBackward0`, `AddBackward0`) and links it to the nodes that produced its inputs. By the time forward finishes, you have one DAG sitting in memory, rooted at the loss, with edges pointing backward toward the leaves (x, w, b).
- **Backward pass**: `.backward()` doesn't construct anything new by default — it just walks the existing DAG from the root toward the leaves, calling each node's `.backward()` method (the local derivative rule) and passing gradients along the same edges that were recorded during forward, just traversed in the opposite direction.

So it's one graph, two passes over it — forward to build it, backward to walk it.

The one case where you do get a second graph is `create_graph=True`. Then the backward computation itself gets tracked with its own `grad_fn`s, producing a new graph on top of the gradients — that's what enables higher-order derivatives (like Hessian-vector products). But that's an opt-in exception, not what happens normally.