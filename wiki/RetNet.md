---
created: "2026-05-20"
title: "Retentive Network: A Successor to Transformer for Large Language Models"
authors: "Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei"
year: 2023
arxiv: "2307.08621"
citation_count: 745
tags: [rnn, linear-attention, retention, efficiency, inference, parallelism, language-model]
tldr: "RetNet introduces a retention mechanism with three equivalent computation forms: parallel (train), recurrent (O(1) inference), and chunkwise (efficient long sequences). Compared to Transformer: 3.4× lower GPU memory, 8.4× lower latency, 15.6× higher throughput — without sacrificing perplexity."
aliases: [RetNet, Retentive Network, RETNET]
theme: efficiency
---

# RetNet: Retentive Network

> Yutao Sun, Li Dong et al. (Microsoft Research), "Retentive Network: A Successor to Transformer for Large Language Models", arXiv:2307.08621 (2023)

## TL;DR

The [[Transformer]] is caught in a "impossible triangle": you can have two of (training parallelism, good performance, low inference cost), but not all three. RNNs have training issues. Linear attention approximations hurt performance.

**RetNet** claims all three. The key is the **retention mechanism** — a linear attention variant with a fixed positional decay that supports three equivalent computational representations:

1. **Parallel form** → train like a Transformer (full matmuls, fully parallel)
2. **Recurrent form** → decode like an RNN ($O(1)$ per step, no KV cache)
3. **Chunkwise recurrent form** → handle long sequences efficiently (linear complexity)

Concrete inference wins vs. Transformer at 8K sequence length: **3.4× lower GPU memory**, **8.4× lower latency**, **15.6× higher throughput** — with equivalent perplexity.

---

## The Core Idea — The Retention Mechanism

RetNet replaces attention with **multi-scale retention (MSR)**. The retention score between query $q_n$ and key $k_m$ is:

$$
\text{Retention}_{n,m} = \gamma^{n-m} (q_n k_m^\top)
$$

where $\gamma \in (0, 1)$ is a **fixed positional decay** (not learned). Older tokens ($m \ll n$) contribute exponentially less. This is the key structural choice: the decay is independent of the token content, making the recurrence time-invariant and analytically tractable.

The output is:

$$
\text{Ret}(X) = (QK^\top \odot D) V
$$

where $D_{nm} = \gamma^{n-m}$ for $n \geq m$, else 0 (causal mask with decay).

### Three computation forms

**Parallel (training):**
$$
\text{Ret}(X) = (QK^\top \odot D) V
$$
Compute the full decayed attention matrix. Parallelizable as a matmul, but $O(T^2)$ memory like Transformers. Used only during training.

**Recurrent (inference):**

Define the recurrent state $S_n = \gamma S_{n-1} + k_n^\top v_n \in \mathbb{R}^{d \times d}$

Then: $\text{ret}(x_n) = q_n S_n$

This is $O(1)$ per step — no growing cache, just one fixed-size matrix.

**Chunkwise (long sequences):**

Chunk the sequence into blocks of size $B$. Within each chunk: apply the parallel form (cheap, $B \ll T$). Between chunks: propagate the recurrent state. Complexity: $O(B \cdot T)$, linear in sequence length.

---

## Key Concepts

- **[[Retention mechanism]]** — RetNet's core: the decayed, causal attention substitute with parallel, recurrent, and chunkwise forms
- **[[Multi-scale retention]] (MSR)** — multiple retention heads with different γ values (like multi-head attention but with different time scales per head)
- **[[Chunkwise recurrent]]** — the third computation mode: intra-chunk parallel + inter-chunk recurrent; enables linear-complexity long-sequence training
- **[[Positional decay]]** — $\gamma^{n-m}$: the fixed exponential downweighting of older tokens. Not learned; instead, $\gamma$ is set per head
- **[[Recurrent state]]** — $S_n \in \mathbb{R}^{d \times d}$: the $O(d^2)$ fixed-size memory that replaces the growing KV cache
- **[[XPOS]]** — a relative positional encoding used with the retention mechanism for better length extrapolation

---

## Architecture / Method

Each RetNet layer replaces multi-head attention with multi-scale retention (MSR):

```
x → MSR(LN(x)) + x → FFN(LN(x)) + x
```

The MSR layer:
- $h$ retention heads with different $\gamma$ values (e.g., $\gamma_i = 1 - 2^{-5 - i}$ for $i=1..h$)
- Each head: $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- Group norm after the retention (replaces attention's softmax normalization)
- Project and concatenate heads

The group norm inside retention replaces the softmax and is critical for training stability — it normalizes the $d \times d$ state accumulations.

---

## Key Results

| Metric | Transformer | RetNet |
|---|---|---|
| Perplexity (1B, 100B tokens) | baseline | **equivalent** |
| GPU memory (8K seq) | baseline | **3.4× lower** |
| Decoding throughput (8K seq) | baseline | **15.6× higher** |
| Decoding latency (8K seq) | baseline | **8.4× lower** |
| Training throughput | baseline | **25–50% slower** |

The training slowdown exists because the parallel retention form is less GPU-efficient than optimized FlashAttention. The inference gains, however, are dramatic because RetNet needs no KV cache at all — just the fixed $d \times d$ state matrix per layer.

Scaling: RetNet shows competitive perplexity vs Transformer at 1.3B, 2.7B, 6.7B parameter scales on the same data.

---

## Comparison to Prior Work

- vs. **[[Transformer]]** — same quality, dramatically cheaper inference. Training is somewhat slower.
- vs. **[[RWKV]]** — very similar conceptually (exponential decay, parallel/recurrent duality). RetNet's decay is also input-independent. Main differences: RetNet adds the chunkwise form, uses group norm instead of layer norm inside the mechanism, and uses multi-scale (different γ per head) vs. RWKV's per-channel decay.
- vs. **[[Mamba]]** — RetNet's decay is fixed (input-independent); Mamba's SSM transition is input-selective. Mamba is better at tasks requiring content-gated recall. RetNet is simpler and easier to implement.
- vs. **[[xLSTM]]** — RetNet's scalar state per head (via $d \times d$ matrix) is similar to xLSTM's mLSTM matrix memory, but without exponential gating.
- vs. **Linear attention** (Performer, Katharopoulos 2020) — RetNet is linear attention with a specific positional decay kernel. Prior linear attention methods degraded significantly vs. softmax attention; RetNet's decay + group norm combination closes most of the gap.

---

## Limitations

- **Input-independent decay** — like RWKV, the decay $\gamma$ doesn't adapt to content. Mamba's selective SSM is more powerful for tasks where the relevant signal is content-dependent.
- **$d \times d$ state per head** — for $d = 256$ per head that's 64K floats per head per layer; scales with model dimension.
- **Training throughput** — 25–50% slower than Transformer due to the parallel retention form being less optimized than FlashAttention; partially addressed by the chunkwise form.
- **No published model above ~7B** in the original paper; scaling beyond that is not well-characterized.

---

## Why It Matters

- **It makes the impossible triangle tractable.** RetNet's three computation forms are the cleanest demonstration that parallelism, performance, and cheap inference can coexist — you just need to train in parallel mode and decode in recurrent mode.
- **It influenced the linear RNN design space.** The combination of parallel/recurrent duality and explicit positional decay directly shaped [[Griffin]] (Google DeepMind, 2024) and influenced the framing in [[Transformers Are SSMs]].
- **It shows that the KV cache is not inevitable.** For production deployment at long contexts, the KV cache is one of the largest cost centers. RetNet's $O(1)$ state shows the architectural alternative exists.

---

## Related Notes

[[RWKV]] · [[Mamba]] · [[xLSTM]] · [[Transformer]] · [[Transformers Are SSMs]] · [[Griffin]] · [[KV Cache]] · [[Linear attention]] · [[State Space Model]]
