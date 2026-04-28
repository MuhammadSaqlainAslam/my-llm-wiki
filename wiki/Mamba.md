---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
authors: "Gu & Dao"
year: 2024
tags: [ssm, efficiency, linear-time, recurrence, selectivity]
tldr: "Make SSM parameters (B, C, Δ) functions of the input so the model selectively compresses context rather than blindly forgetting it. Hardware-aware kernel fusion makes this practical. Matches Transformer quality at 5× inference throughput."
theme: efficiency
---

# Mamba

> Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024

## The Core Tension in Sequence Modeling

Every sequence model makes a tradeoff between **efficiency** and **effectiveness**. Specifically, the question is: how do you compress context into a finite state?

- **Transformers** don't compress at all. They store the entire context (the KV cache) and attend to everything at every step. This is maximally effective but costs $O(n^2)$ in compute and $O(n)$ in memory per step. It doesn't scale to long sequences.
- **RNNs** compress aggressively into a fixed-size hidden state. Constant memory, $O(1)$ per step, but that fixed state can't hold everything you'd want — information gets overwritten.

Mamba's thesis: **the failure mode of prior recurrent models is not that they compress — it's that they compress blindly**. They treat every token equally. What you want is *selective* compression: remember the important stuff, throw away the noise.

---

## State Space Models (SSMs): The Foundation

An SSM maps an input sequence $x(t)$ to output $y(t)$ through a latent state $h(t) \in \mathbb{R}^N$:

$$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
$$y(t) = \mathbf{C} h(t)$$

This is a continuous-time linear system — like a Kalman filter. For deep learning you discretize it. Using zero-order hold with step size $\Delta$:

$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$$
$$\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - I) \cdot \Delta \mathbf{B}$$

Then the discrete recurrence is:
$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t$$
$$y_t = \mathbf{C} h_t$$

This can also be written as a **global convolution** $y = x * \bar{\mathbf{K}}$ where $\bar{\mathbf{K}} = (\mathbf{C}\bar{\mathbf{B}},\ \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\ \ldots)$. The dual representation matters:
- **Train** with convolution (parallelizable, see all tokens at once)
- **Inference** with recurrence (constant state, fast autoregressive generation)

The four parameters $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ define the model. Prior SSMs (S4 and friends) make these **time-invariant** — fixed constants, independent of the input. This is the flaw Mamba fixes.

---

## Why Fixed Parameters Fail

Consider two synthetic tasks that stress-test sequence models:

**Selective Copying:** Given a sequence with colored tokens interspersed with white noise, reproduce only the colored tokens in order. The spacing is random. A time-invariant model can't handle random spacing — it would need a different convolution kernel for each possible spacing. Mamba solves it trivially.

**Induction Heads:** If you've seen "AB" earlier in the sequence, and now you see "A" again, predict "B". This requires associative recall — look up based on content, not just position. Critical for in-context learning. Time-invariant models fail because their dynamics can't depend on what they've seen.

The failure is structural. An LTI (linear time-invariant) model can't selectively remember or forget based on the current input. Making the parameters input-dependent breaks this constraint.

---

## The Selection Mechanism

The fix is elegant: **make $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ functions of the input $x$**.

```
S4 (LTI):      A, B, C, Δ are constants
Mamba (S6):    B, C, Δ = Linear(x)   ← input-dependent
```

Concretely:
- $s_B(x) = \text{Linear}_N(x)$
- $s_C(x) = \text{Linear}_N(x)$
- $s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$, then passed through softplus

The tensor shapes: before selection, $\mathbf{B}$ had shape $(D, N)$ — same for every timestep. After selection, it has shape $(B, L, N)$ — different for every position in the sequence.

**What $\Delta$ does:** It controls how much the model "focuses" on the current input vs. retaining the state. Large $\Delta$ means the state is reset to focus on $x_t$. Small $\Delta$ means the state is preserved and $x_t$ is treated as a transient. This generalizes the gating mechanism of LSTMs and GRUs.

**What $\mathbf{B}$ and $\mathbf{C}$ do:** $\mathbf{B}$ controls whether $x_t$ enters the state at all. $\mathbf{C}$ controls which parts of the state are read out. Together they allow the model to selectively write to and read from memory based on content.

---

## The Hardware Problem (and How They Solve It)

Making $\mathbf{B}, \mathbf{C}, \Delta$ input-dependent breaks the convolution formulation — you can no longer precompute the kernel. You're back to recurrence, which is sequential and memory-intensive.

The naive approach: expand the full state $h$ of shape $(B, L, D, N)$ and process it. This is enormous — $N$ times larger than the input, where $N \approx 16$.

**Mamba's hardware-aware algorithm:**
1. Don't materialize the expanded state in GPU HBM (high-bandwidth memory, the slow part)
2. Load $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ from HBM to fast SRAM
3. Perform the discretization and recurrence entirely in SRAM
4. Write only the final output $(B, L, D)$ back to HBM

This is **kernel fusion** — merge all the operations into one GPU kernel so you avoid repeated HBM reads/writes. Intermediate states never hit HBM.

For backpropagation, instead of saving the intermediate states, they **recompute** them during the backward pass by re-loading the inputs from HBM. Memory usage matches FlashAttention-2.

Result: 40x faster than a naive PyTorch scan implementation. At sequence lengths > 2K, faster than FlashAttention-2.

---

## Mamba Architecture

The Mamba block combines the SSM with a gated MLP:

```
Input x
  ├── Linear (expand D → 2ED)
  │     ├── Branch 1: Conv1D → SSM → [multiply]──┐
  │     └── Branch 2: σ (SiLU gate) ─────────────┘
  └── Linear (contract 2ED → D)
```

This is inspired by the H3 block (SSM + attention) fused with a gated MLP, but simplified so it's homogeneous — the same block stacked repeatedly. They set expansion factor $E = 2$ and use SiLU activation. The inner SSM contributes relatively few parameters; most parameters are in the linear projections.

The Conv1D is a short (kernel size 4) depthwise convolution before the SSM — it provides a small local context window before the global recurrent mixing.

---

## Results

**Language modeling (Pile dataset, Chinchilla scaling protocol):**

Mamba is the first attention-free model to match a strong "Transformer++" recipe (SwiGLU MLP, RMSNorm, rotary embeddings, no bias — the LLaMA baseline). Previous SSMs were all behind.

| Model | Size | Avg zero-shot |
|---|---|---|
| Pythia | 1.4B | 55.2 |
| RWKV | 1.5B | 54.3 |
| **Mamba** | **1.4B** | **59.7** |

Mamba-1.4B beats every model trained the same way and generally matches models at 2-3x its parameter count.

**Inference throughput:** 5x higher than Transformers of the same size (no KV cache needed, constant memory per step).

**Induction heads extrapolation:** Trained at sequence length 256, tested at lengths up to 1M. Mamba achieves perfect accuracy at 1M. Transformers fail beyond 2x training length.

---

## What Mamba Can't Do

Mamba compresses context into a finite state. This is great for most sequence modeling, but it means **exact retrieval** from long contexts is harder than for Transformers. If you need to look up a specific fact from 100K tokens ago, a Transformer with KV cache can find it exactly. Mamba might have overwritten it.

In practice, hybrid architectures solve this: put a few attention layers (for exact recall) alongside many Mamba layers (for cheap sequence processing). This is the approach taken by [[Nemotron-3]].

---

## Connection to Other Architectures

- **Linear attention:** A degenerate SSM where $\mathbf{A} = 1$. Very fast but poor quality.
- **RWKV:** An LTI recurrence with attention-free Transformer structure. Pre-Mamba state of the art among non-attention models.
- **RetNet:** Another linear recurrence approach, similar to RWKV.
- **Hyena:** Uses SSMs parameterized by MLPs (global convolution) — LTI, so no selective memory.

Mamba's key contribution vs. all of these: **selectivity** achieved without sacrificing efficiency via the hardware-aware scan algorithm.

---

## Key Numbers

| Parameter | Value |
|---|---|
| State dimension $N$ | 16 |
| Expansion factor $E$ | 2 |
| Conv1D kernel size | 4 |
| $\Delta$ range (init) | $[0.001, 0.1]$ |
| Mamba-3B perplexity vs. Transformer-3B | Better |
| vs. Transformer-6B | Matches |
| Inference throughput | 5x Transformer |

---

*Related: [[Transformer]] · [[Nemotron-3]] · [[Mixture-of-Experts]]*
