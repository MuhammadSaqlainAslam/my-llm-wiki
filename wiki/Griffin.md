---
created: "2026-05-20"
title: "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models"
authors: "Soham De, Samuel L. Smith, Anushan Fernando et al. (Google DeepMind)"
year: 2024
arxiv: "2402.19427"
citation_count: 254
tags: [rnn, recurrence, local-attention, hybrid, efficiency, inference, gated, google-deepmind]
tldr: "Griffin combines a new gated linear recurrent unit (RG-LRU) with local (sliding-window) attention in a hybrid architecture. At 14B parameters, it matches LLaMA-2 performance while being trained on 7× fewer tokens. Hawk (pure recurrence version) beats Mamba. Inference throughput is 2.5× higher than Transformers at 4K tokens decoded."
aliases: [Griffin, Hawk, RG-LRU]
theme: efficiency
---

# Griffin

> Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev et al. (Google DeepMind), "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", arXiv:2402.19427 (2024)

## TL;DR

Google DeepMind's answer to the question "can a recurrent model match Transformer quality at scale?" Two architectures:

- **Hawk** — pure gated linear recurrence (RG-LRU). No attention at all. Beats reported [[Mamba]] performance at 3B parameters, trained on half as many tokens.
- **Griffin** — hybrid: alternates RG-LRU blocks with **local (sliding-window) attention** blocks. At 7B–14B parameters, matches [[LLaMA 2]] despite seeing 7× fewer training tokens.

The key innovation is the **Real-Gated Linear Recurrent Unit (RG-LRU)**: a diagonal linear recurrence with real-valued (not complex) coefficients and an input gate — simpler than RWKV, more structured than LSTM, and hardware-efficient on TPUs via a custom Pallas kernel.

---

## The Core Idea — RG-LRU: The Simplest Useful Recurrence

The central building block is the RG-LRU layer. Given input $x_t$:

$$
a_t = \sigma(W_a x_t)^2 \cdot \alpha   \quad \text{(input-dependent decay)}
$$

$$
h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \cdot (W_x x_t)   \quad \text{(gated state update)}
$$

$$
y_t = h_t \odot \sigma(W_g x_t)   \quad \text{(output gate)}
$$

Here:
- $h_t \in \mathbb{R}^d$ is the recurrent state (a **vector** — not a matrix like xLSTM's mLSTM)
- $a_t$ is an **input-dependent decay**: a sigmoid-squared of a linear projection, multiplied by a fixed per-channel scalar $\alpha$ initialized near 1. The squared sigmoid keeps $a_t \in [0, 1]$, ensuring stability.
- The $\sqrt{1 - a_t^2}$ factor is a **normalization** that preserves the norm of $h_t$ over time (keeping gradients healthy)
- $\sigma(W_g x_t)$ is an output gate that further filters the state

The input-dependence of $a_t$ is critical: unlike RWKV and RetNet (where decay is fixed per channel), RG-LRU's decay changes based on the current input. This gives the model content-awareness without going all the way to Mamba's fully input-selective SSM.

---

## Key Concepts

- **[[RG-LRU]]** (Real-Gated Linear Recurrent Unit) — Griffin's core recurrent layer; diagonal linear recurrence with input-gated decay
- **[[LRU]]** (Linear Recurrent Unit) — the predecessor to RG-LRU (Orvieto et al., 2023); complex-valued diagonal recurrence; RG-LRU is the real-valued, input-gated extension
- **[[Local attention]]** — sliding-window attention that attends only to the last $W$ tokens; $O(W \cdot T)$ instead of $O(T^2)$; Griffin uses $W = 1024$
- **[[Hawk]]** — the pure-recurrence version of the Griffin architecture: only RG-LRU blocks, no attention layers
- **[[Pallas kernel]]** — Google's JAX kernel language (like Triton for TPUs); used to implement the RG-LRU scan without materializing HBM intermediates (analogous to FlashAttention's IO-aware approach)
- **[[Diagonal recurrence]]** — $h_t = a_t \odot h_{t-1} + \ldots$: the state transition matrix is diagonal (independent channels), enabling full parallelism across the state dimension and efficient hardware utilization

---

## Architecture / Method

Both Hawk and Griffin use the same overall structure: residual blocks alternating between a **temporal mixing block** and an **MLP block**.

### Hawk (pure recurrence)
```
x → LN → RG-LRU block → + x
x → LN → MLP (gated GELU) → + x
```

### Griffin (hybrid)
```
Every 3 blocks:
  2× [LN → RG-LRU block → + x] then [LN → MLP → + x]
  1× [LN → Local-MQA (window=1024) → + x] then [LN → MLP → + x]
```

The 2:1 ratio of recurrent-to-attention blocks is tuned empirically. Local attention gives Griffin the ability to do exact short-range lookup (within 1024 tokens), while the RG-LRU accumulates information from the full history.

Griffin-14B has 46 layers: ~30 RG-LRU blocks and ~16 local attention blocks (plus the interleaved MLPs).

---

## Key Results

| Model | Params | Tokens trained | Performance |
|---|---|---|---|
| **Hawk-3B** | 3B | 300B | Beats Mamba-3B (same benchmark, 2× fewer tokens) |
| **Griffin-7B** | 7B | 300B | Matches LLaMA-2-7B (trained on 7× more tokens) |
| **Griffin-14B** | 14B | 300B | Matches LLaMA-2-13B |
| Hawk (any) | any | — | 2.5× throughput vs MQA Transformer at 1024+ decoded tokens |
| Griffin (any) | any | — | Higher throughput than Transformer; lower than Hawk |

Scaling: both Hawk and Griffin show **power-law scaling** (held-out loss vs. FLOPs), meaning they benefit from more compute predictably — not a fluke of small-scale tuning.

Length extrapolation: Griffin generalizes to sequences longer than training length (tested up to 4× training length), while standard Transformer performance degrades.

---

## Comparison to Prior Work

- vs. **[[Mamba]]** — Hawk beats Mamba-3B at the same scale with fewer training tokens. Key difference: RG-LRU is simpler than Mamba's selective SSM (no $\Delta$, $B$, $C$ per token), but adds an **input gate** that Mamba lacks. Both are diagonal recurrences.
- vs. **[[RWKV]]** — RWKV's WKV mechanism uses channel-wise fixed decay; RG-LRU's decay is input-dependent. Griffin adds local attention; RWKV doesn't. Griffin scales better at the 14B level.
- vs. **[[RetNet]]** — RetNet has fixed decay; RG-LRU is input-gated. RetNet explicitly has the parallel/recurrent duality; Griffin's RG-LRU also admits a parallel training mode via the associativity of the linear scan.
- vs. **[[xLSTM]]** — xLSTM's mLSTM has a matrix state ($d \times d$); RG-LRU has a vector state ($d$). xLSTM is more expressive; RG-LRU is faster.
- vs. **[[LLaMA 2]]** — Griffin matches LLaMA 2 performance at the same scale while seeing 7× fewer training tokens. This is remarkable sample efficiency.

---

## Limitations

- **Vector state only.** $h_t \in \mathbb{R}^d$ is a fixed-size vector — no associative recall like xLSTM's matrix memory. For exact key-value lookup over the full context, local attention windows are the fallback.
- **Retrieval within local window only.** The sliding-window attention covers 1024 tokens; anything older must be compressed into the RG-LRU state, which is lossy.
- **TPU-centric implementation.** The efficient RG-LRU kernel was written in Pallas (JAX/TPU). CUDA implementations came later from the community.
- **Not yet at GPT-4 scale.** The paper's largest model is 14B; behavior at 70B+ is uncharted.

---

## Why It Matters

- **Google DeepMind's validation of the hybrid approach.** When a top-tier lab publishes results showing recurrent models match Transformers at 14B with 7× less data, the field takes notice.
- **RG-LRU is the simplest input-gated recurrence that works at scale.** It's more structured than Mamba, simpler than xLSTM, and hardware-friendly. It represents a design point worth knowing.
- **It connects to Gemini's architecture.** Later Google models reportedly use Griffin-inspired hybrid architectures; the RG-LRU pattern appears in production systems.
- **It demonstrates that local attention + global recurrence is a stable hybrid recipe.** The 2:1 recurrent/attention ratio in Griffin is a practical template for anyone building a hybrid model.

---

## Related Notes

[[Mamba]] · [[RetNet]] · [[RWKV]] · [[xLSTM]] · [[LLaMA 2]] · [[Transformer]] · [[State Space Model]] · [[KV Cache]] · [[Local attention]] · [[Nemotron-3]]
