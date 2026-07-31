---
created: "2026-05-20"
title: "RWKV: Reinventing RNNs for the Transformer Era"
authors: "Bo Peng et al. (EleutherAI community)"
year: 2023
arxiv: "2305.13048"
citation_count: 1150
tags: [rnn, linear-attention, efficiency, inference, language-model, recurrence, parallelism]
tldr: "RWKV replaces dot-product attention with a linear attention variant that can be computed either as a parallelizable Transformer (training) or as an O(1)-per-step RNN (inference). Scales to 14B parameters — the largest dense RNN ever trained — while matching GPT-NeoX-20B quality at a fraction of inference cost."
aliases: [RWKV, "RWKV-7", "Receptance Weighted Key Value"]
theme: efficiency
---

# RWKV: Reinventing RNNs for the Transformer Era

> Bo Peng, Eric Alcaide, Quentin Anthony et al. (EleutherAI), "RWKV: Reinventing RNNs for the Transformer Era", EMNLP 2023 (arXiv:2305.13048)

## TL;DR

The [[Transformer]] has two great properties: training is parallelizable (no token-to-token sequential dependency) and any two tokens can interact directly. It has one great flaw: inference is $O(T^2)$ memory and the [[KV Cache]] grows with context length.

RNNs flip this: $O(1)$ inference per step, constant memory, but training is sequential and long-range gradients vanish.

**RWKV** (Receptance Weighted Key Value) finds the sweet spot: a linear attention formulation that looks exactly like a Transformer during training (matmul, fully parallel) but degenerates into a time-invariant RNN recurrence during inference. No approximation, no architecture switch — the same parameters, two different computation modes.

Crucially, RWKV avoids the quadratic cost of attention by replacing the softmax with a scalar-weighted decay: tokens "attend" to past context through an exponentially decaying sum rather than a full dot-product. This is O(Td) at training, O(d) at inference — the best of both.

---

## The Core Idea — Linear Attention as an RNN

Standard attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

This requires materializing $T \times T$ scores. RWKV replaces softmax with a positional decay:

$$
\text{WKV}_t = \frac{\sum_{i < t} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t}{\sum_{i < t} e^{-(t-1-i)w + k_i} + e^{u + k_t}}
$$

Here:
- $w \geq 0$ is a **learned channel-wise decay** — older tokens get exponentially downweighted
- $u$ is a **learned bias** for the current token (lets the model "focus" on the present)
- $k_t, v_t$ are per-token key/value vectors

This WKV operation has a numerator and denominator that can be computed **incrementally**:

$$
a_t = e^{-w} \cdot a_{t-1} + e^{k_t} v_t, \quad b_t = e^{-w} \cdot b_{t-1} + e^{k_t}
$$

So at inference, one step requires only $a_t, b_t$ from the previous step — constant memory, $O(d)$ computation.

At training, the full sum can be computed as a **cumulative product + scan** that is parallelizable across time, just like FlashAttention tiles the attention sum.

---

## Key Concepts

- **[[Receptance]]** — RWKV's gating vector $r_t$: controls how much the output uses the WKV signal vs. being zero. Analogous to an LSTM forget gate.
- **[[Time-mixing]]** — the RWKV layer that implements the WKV attention (cross-time interactions)
- **[[Channel-mixing]]** — a pointwise FFN-like transformation in each RWKV block (within-position)
- **[[Exponential decay]]** — the $e^{-w}$ factor that downweights old tokens; $w$ is learned per channel and must be $\geq 0$ (enforced by reparameterization)
- **[[Time shift]]** — each layer mixes the current token's embedding with the *previous* token's embedding via a learned interpolation: $\hat{x}_t = \alpha x_t + (1-\alpha) x_{t-1}$. This cheap operation gives each layer a local context window.
- **[[Linear attention]]** — the general family of attention approximations where the softmax is replaced with a kernel; RWKV is a special case with a specific decay kernel

---

## Architecture / Method

An RWKV block stacks two sublayers:

```
x_t
 │
 ├─→ Time-mixing (WKV + receptance gating) → residual
 │
 └─→ Channel-mixing (pointwise FFN) → residual
```

**Time-mixing** in full:
```
r_t = W_r · (μ_r x_t + (1-μ_r) x_{t-1})    # receptance
k_t = W_k · (μ_k x_t + (1-μ_k) x_{t-1})    # key
v_t = W_v · (μ_v x_t + (1-μ_v) x_{t-1})    # value
wkv_t = WKV(r, k, v, w, u)
o_t  = W_o · (σ(r_t) ⊙ wkv_t)
```

**Channel-mixing** is a gated FFN:
```
r'_t = W_r' · (μ x_t + (1-μ) x_{t-1})
k'_t = W_k' · (μ x_t + (1-μ) x_{t-1})
o'_t = σ(r'_t) ⊙ (W_v' · max(k'_t, 0)²)
```

The full model stacks $N$ such blocks with LayerNorm. It looks like a Transformer — same block structure, same parameter count — but the time-mixing layer is the WKV recurrence instead of multi-head attention.

---

## Key Results

| Model | Params | Benchmark | Notes |
|---|---|---|---|
| RWKV-4 | 14 B | Pile validation | First dense RNN at 14B scale |
| RWKV-4 | 14 B | NLP benchmarks | Matches GPT-NeoX-20B on most tasks |
| RWKV (any) | any | Inference | **O(1) memory** per step; no KV cache needed |
| Inference cost | — | vs Transformer | Linear in sequence length, not quadratic |

Complexity comparison:

| Model | Time (inference) | Space (inference) |
|---|---|---|
| Transformer | $O(T^2 d)$ | $O(T^2 + Td)$ |
| Linear Transformer | $O(Td^2)$ | $O(Td + d^2)$ |
| **RWKV** | **O(Td)** | **O(d)** |

---

## Comparison to Prior Work

- vs. **[[Transformer]]** — RWKV matches Transformer quality at 1–14B scale on language tasks; inference is dramatically cheaper (no KV cache, $O(1)$ step memory).
- vs. **[[LSTM]]** — same $O(1)$ inference, but RWKV trains in parallel (unlike LSTM's sequential dependency). No hidden-to-hidden weight matrix: the time-mixing uses only $W_k, W_v, W_r$ on the *input* at each step.
- vs. **[[Mamba]]** — both are linear-time recurrences. Key difference: RWKV's decay $w$ is **fixed per channel** (input-independent); Mamba's transition matrices are **input-selective** (data-dependent). This makes Mamba better at recall tasks where the gate should depend on content.
- vs. **RetNet** — RetNet also uses exponential decay; RWKV and RetNet are very similar in spirit. RWKV's time-shift mixing and the community-driven 14B scale distinguish it in practice.
- vs. **[[xLSTM]]** — xLSTM uses exponential gating (like RWKV) but adds a matrix memory cell; RWKV uses only a scalar recurrent state per channel.

---

## Limitations

- **Input-independent decay.** $w$ doesn't change based on content, only position. The model can't selectively "snap awake" for a surprising token the way [[Mamba]]'s selective SSM can.
- **No content-based retrieval.** Because the exponential decay replaces the full dot-product, RWKV cannot implement exact key-value lookup — for needle-in-haystack tasks at long range, it underperforms attention.
- **Recall degrades gracefully, not sharply.** Long-distance information is exponentially downweighted; the model can still work on most practical tasks but fails on tasks requiring verbatim recall of distant tokens.
- **Community-trained models** — the 14B RWKV model was trained by the EleutherAI community, not an industrial lab. Evaluation is less rigorous than comparable Transformer releases.

---

## Why It Matters

- **It proves you can scale linear RNNs to 14B parameters.** Before RWKV, the practical ceiling for RNN-style models was in the hundreds of millions. RWKV showed scaling works.
- **It's a clean existence proof of the training/inference duality.** The same parameters run as a Transformer (parallel, matmul-heavy) or as an RNN (sequential, $O(1)$). This insight directly influenced how later models (RetNet, Griffin) frame their dual computation modes.
- **It's genuinely open.** RWKV was trained and released by a community collaboration, with all weights and code public. This gave the SSM/linear-RNN research community a baseline to build on.

---

## Related Notes

[[Mamba]] · [[xLSTM]] · [[RetNet]] · [[Transformer]] · [[LSTM]] · [[State Space Model]] · [[KV Cache]] · [[Linear attention]]
