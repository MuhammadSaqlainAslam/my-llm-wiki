---
created: "2026-05-22"
title: micrograd
tags: [autograd, backpropagation, deep learning, neural network, educational, scalar]
year: 2020
tldr: A ~150-line scalar-valued autograd engine implementing reverse-mode autodiff over a dynamically built DAG, plus a tiny PyTorch-like neural network library on top — ideal for understanding backpropagation from first principles.
wikilinks: ["[[nanoGPT]]", "[[llm.c]]"]
---

# micrograd

**Repository:** [karpathy/micrograd](https://github.com/karpathy/micrograd)
**Author:** Andrej Karpathy
**Year:** 2020
**License:** MIT

## Overview

micrograd is an educational scalar autograd engine — essentially the smallest possible implementation of the core machinery that powers all modern [[Transformer|deep learning]] frameworks. It implements **reverse-mode automatic differentiation** (backpropagation) over a dynamically constructed directed acyclic graph (DAG).

The entire engine is ~100 lines of Python; the neural network library built on top is ~50 lines.

## Architecture

### Engine (`engine.py`)
- `Value` class wraps a scalar float and tracks its computational history
- Supports: `+`, `-`, `*`, `/`, `**`, `relu`, `tanh`, `exp`
- `.backward()` runs topological sort and applies chain rule recursively

### Neural Network Library (`nn.py`)
- `Neuron` — single neuron with weights, bias, and activation
- `Layer` — a list of Neurons
- `MLP` — multi-layer perceptron stacking Layers

## Example

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
g = (c - d)**2 / 2.0
g.backward()

print(a.grad)  # dg/da
print(b.grad)  # dg/db
```

## Why It Matters

| Concept | Illustration in micrograd |
|---|---|
| Forward pass | Python operator overloading builds the DAG |
| Backward pass | `_backward` closures store local gradient functions |
| Chain rule | Topological traversal accumulates `.grad` |
| Dynamic graph | DAG rebuilt every forward pass (like PyTorch eager mode) |

## Relation to Wiki Themes

micrograd is the conceptual foundation beneath all [[Transformer|neural network]] training. Understanding it is prerequisite to understanding how [[transformer]] training, [[RLHF]], and other gradient-based methods work. See [[nanoGPT]] for a full GPT training stack, or [[llm.c]] for a C/CUDA implementation without any autograd framework.
