---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
authors: "DeepSeek-AI"
year: 2026
tags: [moe, long-context, attention, compression, sparse-attention, muon, post-training, rl]
tldr: "Crack the quadratic attention bottleneck for 1M-token contexts by replacing standard attention with two new compressed variants (CSA + HCA), upgrading residual connections with manifold-constrained hyper-connections, and using the Muon optimizer. Result: at 1M tokens, DeepSeek-V4-Pro uses only 27% of DeepSeek-V3's FLOPs and 10% of its KV cache."
theme: efficiency
---

# DeepSeek-V4

> DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", 2026

## The Problem: Attention Doesn't Scale to 1M Tokens

[[Transformer]] attention is $O(n^2)$ in sequence length. At 1M tokens that's $10^{12}$ operations per layer — **not viable**. But long contexts are exactly where the next wave of capability lives: multi-document reasoning, complex agentic workflows, and test-time scaling (letting the model think for thousands of tokens before answering).

DeepSeek-V3.2 could nominally handle long contexts but the cost was prohibitive. DeepSeek-V4's entire architecture is organized around one goal: **make a million tokens cheap**.

The two models in the series:
- **DeepSeek-V4-Pro**: 1.6T total parameters, 49B activated per token
- **DeepSeek-V4-Flash**: 284B total parameters, 13B activated per token

Both trained on 32–33T tokens and natively support 1M-token contexts.

---

## Architecture Overview

DeepSeek-V4 keeps the [[Transformer]] backbone and [[Mixture-of-Experts]] FFN layers from DeepSeek-V3, but makes three targeted upgrades:

1. **Hybrid [[Compressed Sparse Attention|CSA]] + [[Heavily Compressed Attention|HCA]]** — replaces standard attention with two compressed variants
2. **[[Manifold-Constrained Hyper-Connections]] (mHC)** — upgrades residual connections
3. **[[Muon Optimizer|Muon]] optimizer** — replaces AdamW for most parameters

Everything else (MTP modules, DeepSeekMoE, rotary embeddings) is inherited unchanged.

---

## Key Idea 1: Compressed Attention (CSA and HCA)

Standard attention stores one KV entry per token. For 1M tokens that's a 40+ GB [[KV Cache]] and an $O(n^2)$ attention matrix. The fix: **compress many tokens into one KV entry**, then attend over the compressed sequence.

DeepSeek-V4 layers alternate between two compression regimes:

### Compressed Sparse Attention (CSA)

Think of CSA as: *compress aggressively, then be selective about which blocks to attend to*.

**Step 1 — Compress.** Every $m$ consecutive tokens get weighted-summed into a single KV entry using learned positional biases. The sequence shrinks from length $n$ to $n/m$.

**Step 2 — Sparse selection (Lightning Indexer).** Not every compressed block matters for every query. CSA uses a cheap indexer mechanism (low-rank query projections → dot product → top-k) to pick only the $k$ most relevant compressed blocks.

**Step 3 — MQA.** Attend over the selected $k$ compressed blocks using Multi-Query Attention (shared key/value across heads).

**Step 4 — Sliding window supplement.** The most recent $n_\text{win}$ tokens are always attended to in full (uncompressed), because recent tokens are almost always highly relevant. This is the "don't throw away your neighbors" fix.

The compression rate times sparse selection means: instead of attending to $n$ tokens, you attend to roughly $n/(m \cdot \text{sparsity})$ entries. At 1M tokens this is a 10–100x reduction.

### Heavily Compressed Attention (HCA)

CSA still does sparse selection, which has overhead. HCA goes further: **compress so hard you can do dense attention over the result cheaply**.

HCA uses a compression rate $m' \gg m$ (no sparse selection needed — the compressed sequence is tiny). It's dense attention but over a very short sequence. Same sliding window supplement as CSA.

The tradeoff is accuracy vs. speed: HCA is cheaper but coarser. DeepSeek-V4 **interleaves** CSA and HCA layers, so you get the fidelity of CSA on some layers and the extreme efficiency of HCA on others.

### How much does it help?

At 1M-token context vs. DeepSeek-V3.2:

| Model | FLOPs (relative) | KV Cache (relative) |
|---|---|---|
| DeepSeek-V3.2 | 1× | 1× |
| DeepSeek-V4-Pro | **0.27×** | **0.10×** |
| DeepSeek-V4-Flash | **0.10×** | **0.07×** |

A 10× KV cache reduction means you can actually serve these contexts at scale.

---

## Key Idea 2: Manifold-Constrained Hyper-Connections (mHC)

Standard residual connections are `output = layer(x) + x`. Hyper-Connections (HC) expand this: instead of a scalar skip, you maintain a *width-expanded* residual stream of shape $n_\text{hc} \times d$ and learn how to mix it across layers.

**Why bother?** It gives the model a richer "highway" for gradients and information flow across layers without increasing the hidden size of the attention/FFN layers themselves.

**The problem with naive HC:** When stacked deep, numerical instability accumulates. Training frequently diverges.

**mHC's fix:** Constrain the residual mixing matrix $B_l$ to the **Birkhoff polytope** — the set of doubly stochastic matrices (rows and columns sum to 1, all entries $\geq 0$). The key property: any doubly stochastic matrix has spectral norm $\leq 1$, so the residual transform is *non-expansive*. Signals can't blow up.

The projection onto this manifold is done via **Sinkhorn-Knopp** iterations: apply softmax alternately over rows and columns until convergence.

The parameters ($A_l$, $B_l$, $C_l$) for pre-block mixing, residual mixing, and post-block mixing are generated *dynamically* — they depend on the input at each token position. This is computed via cheap low-rank projections.

In practice: mHC improves model quality and eliminates the instability that plagued naive HC, making it practical to stack deep with this mechanism.

---

## Key Idea 3: Muon Optimizer

Adam updates $W \leftarrow W - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$. The division by the second moment normalizes each parameter independently. **Muon** instead approximately orthogonalizes the gradient matrix before applying it — the update is closer to the steepest descent direction in the space of matrices rather than element-wise coordinate descent.

Concretely, Muon applies **Newton-Schulz iterations** to the momentum-updated gradient $G$ to approximate $G \cdot (G^T G)^{-1/2}$ (i.e., the polar factor of $G$). The iterations are:

$$M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^T) M_{k-1} + c (M_{k-1} M_{k-1}^T)^2 M_{k-1}$$

DeepSeek uses a **hybrid schedule**: 8 steps with aggressive coefficients to drive rapid convergence, then 2 steps with stabilizing coefficients to lock singular values at 1.

**Why it works:** The orthogonalized update has a more isotropic effect across the weight matrix. Loss landscapes are typically ill-conditioned; Muon adapts to the geometry more naturally than Adam's coordinate-wise scaling.

**Where it's applied:** Most linear layers. AdamW is kept for embeddings, prediction heads, RMSNorm weights, and the static biases/gates of mHC — places where the geometric interpretation doesn't apply cleanly.

Result: faster convergence and less need to babysit learning rate schedules.

---

## Mixture-of-Experts: Continued from V3

DeepSeek-V4 keeps [[Mixture-of-Experts]] via DeepSeekMoE: fine-grained routed experts + always-on shared experts. A few tweaks from V3:

- Affinity score activation changed from `Sigmoid(·)` to `Sqrt(Softplus(·))` for better gradient properties
- Initial transformer blocks use **Hash routing** (token ID → expert assignment deterministically) instead of learned routing — avoids the cold-start routing collapse problem
- Auxiliary-loss-free load balancing with a light sequence-wise balance loss to prevent extreme within-sequence imbalance
- Expert parameters stored in **FP4** precision (routed experts only) — 2x memory reduction, forward-compatible with future hardware that supports FP4 natively

Combined with FP4 Quantization-Aware Training (QAT), this shrinks the memory footprint significantly and enables larger batch sizes during inference.

---

## Pre-Training

**Data:** 32T (Flash) and 33T (Pro) tokens of diverse, high-quality text. DeepSeek has historically been tight-lipped about data sourcing; the main callout is that the pipeline was redesigned around long-context quality.

**Instability fixes:** FP4 QAT introduces gradient noise. They mitigate with careful loss scaling and by falling back to higher precision for sensitive modules.

**Evaluation:** DeepSeek-V4-Flash-Base already outperforms DeepSeek-V3.2-Base on most benchmarks despite fewer activated parameters — a testament to the architectural efficiency improvements. V4-Pro-Base extends this further.

[[Multi-Token Prediction]] modules are retained unchanged from V3 — the strategy was already validated there, so no changes needed.

---

## Post-Training: Specialist-Then-Distill

The post-training pipeline is a two-stage paradigm:

**Stage 1 — Grow specialists.** For each domain (math, coding, agents, instruction following), train a *separate* expert model:
1. SFT on high-quality domain data to build the foundation
2. RL via **[[GRPO]]** (Group Relative Policy Optimization, from [[RLVR]]) with domain-specific reward models

This avoids the classic multi-task RL conflict where improving one skill hurts another.

**Stage 2 — Distill into one model.** A single unified model is trained via **[[On-Policy Distillation]]** (OPD): the student generates rollouts, then optimizes the reverse KL against the ensemble of teacher specialists. The student learns to balance all the specialists' behaviors in one set of weights.

This is cheaper than keeping multiple deployed models and avoids the quality degradation of simple merging (model soups).

---

## Benchmark Results

| Benchmark | DeepSeek-V4-Pro-Max | Claude Opus 4.6 Max | Gemini 3.1 Pro High |
|---|---|---|---|
| SimpleQA | 57.9 | 46.2 | 45.3 |
| HLE (Pass@1) | 75.6 | 37.7 | 40.0 |
| SWE-Verified | 80.6 | 80.8 | 80.6 |
| Codeforces Rating | 3206 | 3168 | 3052 |

Long-context: V4-Pro-Max surpasses Gemini-3.1-Pro on academic 1M-token benchmarks. Agent tasks: on par with leading open-source models (Kimi-K2.6, GLM-5.1) but slightly below frontier closed models.

**V4-Flash vs. V4-Pro:** Flash closes the gap on reasoning tasks when given a larger thinking budget but falls short on knowledge-heavy benchmarks due to fewer parameters. For cost-sensitive deployments it's a strong choice.

---

## What This Means for the Architecture Landscape

```
Vanilla Attention: O(n²) compute, O(n) KV cache
  ↓ problem at 1M tokens
CSA: compress m tokens → 1, then sparse top-k selection
HCA: compress m' tokens → 1, dense attention over tiny sequence
  → both keep local context via sliding window

Standard residuals: x + layer(x)
  ↓ mHC: expand residual stream to n_hc × d
  → doubly stochastic mixing matrix (spectral norm ≤ 1)
  → stable deep stacking

AdamW: element-wise second-moment scaling
  ↓ Muon: orthogonalize the update matrix via Newton-Schulz
  → faster convergence, better conditioning
```

The common thread: every innovation is motivated by a specific bottleneck — quadratic attention cost, residual signal degradation, optimizer inefficiency — and the fix is mathematically principled rather than heuristic.

---

## Limitations and Open Questions

- **FP4 on today's hardware:** The peak FLOPs for FP4×FP8 currently match FP8×FP8 — the memory savings are real but the compute advantage is a future bet on hardware support
- **mHC overhead:** Dynamic parameter generation adds compute; the paper argues it's small relative to the gains, but it complicates implementation
- **Routing quality at scale:** Hash routing in early layers works empirically but removes the adaptivity that makes MoE powerful — whether this is a net win at larger scale is open
- **OPD fidelity:** Reverse KL distillation mode-seeks; the student may under-represent specialist tails (rare but important capabilities)

---

## Key Numbers

| Property | DeepSeek-V4-Pro | DeepSeek-V4-Flash |
|---|---|---|
| Total parameters | 1.6T | 284B |
| Activated per token | 49B | 13B |
| Context length | 1M tokens | 1M tokens |
| Pre-training tokens | 33T | 32T |
| FLOPs vs V3.2 at 1M ctx | 27% | 10% |
| KV cache vs V3.2 at 1M ctx | 10% | 7% |

---

*Related: [[Transformer]] · [[Mixture-of-Experts]] · [[KV Cache]] · [[GQA]] · [[Multi-Token Prediction]] · [[RLVR]] · [[NVFP4]] · [[Speculative Decoding]] · [[Nemotron-3]]*
