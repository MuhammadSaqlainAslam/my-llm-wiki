---
title: "Karpathy Index — LLM Wiki Master Hub"
tags: [index, meta]
tldr: "All papers, grouped by theme, with a concept glossary and a paragraph on how everything connects."
---

# Karpathy Index

> Every note in one place. Start here if you want the big picture.

---

## Theme I — Foundations

The paper that started everything. Everything else in this wiki is either built on it or reacting to it.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Transformer]] | Attention Is All You Need | 2017 | Self-attention replaces recurrence; any two tokens connect in one step, enabling parallel training and O(1) path length between positions. |

**Tags:** `foundational` `attention` `architecture` `parallelism`

---

## Theme II — Efficient Sequence Modeling

The O(n²) attention cost is fine for short sequences. For long ones it breaks. These papers replace attention with something cheaper.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Mamba]] | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2024 | Make SSM parameters input-dependent so the model selectively compresses context. Hardware-aware kernel fusion makes it practical. 5× inference throughput over Transformers. |

**Tags:** `ssm` `efficiency` `linear-time` `recurrence` `selectivity`

---

## Theme III — Scaling / Parameter Efficiency

Parameters and compute don't have to scale together. These papers decouple them.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Mixture-of-Experts]] | Switch Transformers / Mixtral of Experts | 2022 | Route each token to a sparse subset of expert FFNs; parameters scale cheaply while per-token compute stays constant. More parameters, same FLOPs. |

**Tags:** `scaling` `moe` `routing` `efficiency` `sparse`

---

## Theme IV — Modern Synthesis

What happens when you combine the three themes above into a single production system.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Nemotron-3]] | Nemotron 3: Efficient and Open Intelligence | 2025 | Hybrid Mamba-2 + sparse attention + LatentMoE, trained at NVFP4. 120B/12B active. 7.5× throughput over Qwen3.5-122B at 1M context with competitive accuracy. |
| [[DeepSeek_V4]] | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | 2026 | CSA + HCA compressed attention yields 10× KV cache reduction at 1M tokens. mHC residual connections + Muon optimizer. 1.6T/49B active. SOTA open model. |

**Tags:** `hybrid` `moe` `mamba` `production` `inference` `quantization` `rl` `agentic`

---

## Concept Glossary

Every `[[wikilink]]` that appears anywhere in this wiki, with a one-line definition.

| Concept | What it is | First introduced |
|---|---|---|
| [[Transformer]] | Self-attention-based encoder-decoder; the universal backbone of modern NLP | Vaswani et al. 2017 |
| [[Mamba]] | Selective SSM: input-dependent B, C, Δ parameters enable content-aware compression | Gu & Dao 2024 |
| [[Mixture-of-Experts]] | Sparse routing: each token activates only K of N expert FFNs; decouples params from FLOPs | Fedus et al. 2022 |
| [[Nemotron-3]] | Hybrid Mamba-Transformer-LatentMoE; synthesizes Themes I–III into a production system | NVIDIA 2025 |
| [[DeepSeek_V4]] | MoE LLM with CSA+HCA compressed attention, mHC residual connections, Muon optimizer; 1M-token context at 27% of V3's FLOPs | DeepSeek-AI 2026 |

**Stub pages** — key sub-concepts with their own notes:

| Term | Brief definition | Appears in |
|---|---|---|
| [[KV Cache]] | Stored key-value pairs from attention; grows linearly with sequence length | [[Transformer]], [[Mamba]], [[Nemotron-3]] |
| [[State Space Model]] | Continuous-time linear system discretized for sequences: h_t = Āh_{t-1} + B̄x_t | [[Mamba]] |
| [[Hardware-Aware Scan]] | Kernel fusion keeping SSM recurrence in SRAM; avoids HBM reads for intermediate states | [[Mamba]] |
| [[Load Balancing Loss]] | Auxiliary loss encouraging uniform token distribution across experts | [[Mixture-of-Experts]], [[Nemotron-3]] |
| [[LatentMoE]] | Project d → ℓ before routing; cuts all-to-all communication by d/ℓ, allows more experts | [[Nemotron-3]], [[Mixture-of-Experts]] |
| [[Multi-Token Prediction]] | Auxiliary heads predict 2, 3… tokens ahead; richer training signal + free speculative decoding | [[Nemotron-3]] |
| [[NVFP4]] | NVIDIA 4-bit float (E2M1 elements, 16-element micro-blocks); 3× peak throughput vs BF16 | [[Nemotron-3]] |
| [[Speculative Decoding]] | Draft tokens verified in one parallel pass; [[Multi-Token Prediction]] heads provide drafts | [[Nemotron-3]] |
| [[GQA]] | Grouped-query attention; 32 Q-heads share 2 KV-heads, cutting [[KV Cache]] size 16× | [[Nemotron-3]] |
| [[RLVR]] | RL with verifiable rewards across 21 simultaneous environments (math, code, tools, …) | [[Nemotron-3]] |

**Inline concepts** — too granular for their own page, defined where they first appear:

| Term | Brief definition | Appears in |
|---|---|---|
| Self-attention | Dot-product of Q, K, V matrices; every token attends to every other in one step | [[Transformer]] |
| Multi-head attention | Run h attention heads in parallel, each learning different relationship types | [[Transformer]] |
| Selective SSM (S6) | SSM where B, C, Δ depend on the current input x (vs. fixed constants in S4) | [[Mamba]] |
| Top-K routing | Router selects the K highest-scoring experts per token; all others contribute zero | [[Mixture-of-Experts]] |
| Expert capacity | Max tokens an expert processes per batch; overflow tokens skip the layer via residual | [[Mixture-of-Experts]] |
| NVFP4 PTQ | Post-training quantization to NVFP4 via AutoQuantize NAS-style sensitivity search | [[Nemotron-3]] |
| WSD schedule | Warmup–Stable–Decay LR schedule; stable phase supports checkpoint merging | [[Nemotron-3]] |
| Checkpoint merging | Weighted average of checkpoints from different training windows; 2–4 point free quality gain | [[Nemotron-3]] |
| LC-Phase | Continued pre-training at 1M context (34B tokens) enabling million-token inference | [[Nemotron-3]] |
| PivotRL | Reuses offline SFT traces; focuses RL updates on high-uncertainty "pivot" turns | [[Nemotron-3]] |
| Async GRPO | Group Relative Policy Optimization with asynchronous rollout generation | [[Nemotron-3]] |
| Mamba SSM cache | Recurrent state cache; quantized to FP16 with stochastic rounding to prevent verbosity | [[Nemotron-3]], [[Mamba]] |

---

## How These Papers Connect

The Transformer (2017) solved sequential training by replacing RNNs with self-attention, achieving O(1) path length between any two tokens — but at the cost of O(n²) compute and a KV cache that grows without bound. That single limitation became the organizing constraint for the next eight years of research.

Mamba (2024) attacks the O(n²) problem directly: it shows that the failure mode of older recurrent models was not compression itself but *blind* compression. Make the SSM parameters depend on the input and you get selective memory — the model decides what to keep and what to forget based on content, not just position. A hardware-aware kernel fusion pass makes this as fast in practice as FlashAttention. Mamba matches Transformer quality at 5× throughput and constant memory per step.

Mixture-of-Experts attacks a different axis: not compute per token but the coupling between parameter count and compute. A dense model can't add knowledge without adding FLOPs. MoE breaks that coupling — route each token to K of N experts, so total parameters scale with N while active compute scales only with K. Switch Transformers prove that top-1 routing works; Mixtral proves that top-2 of 8 SwiGLU experts outperforms LLaMA-2 70B at 5× fewer active parameters.

Nemotron-3 (2025) is the synthesis. It takes Mamba's insight (most sequence processing is cheap recurrence; exact recall needs occasional attention) and MoE's insight (parameters and compute are independent axes), adds LatentMoE to cut the all-to-all communication bottleneck by routing in a compressed latent space, trains the whole thing at NVFP4 precision for 3× hardware throughput, uses multi-token prediction to densify the training signal and enable speculative decoding at inference, and then runs 21-environment simultaneous RL to specialize the model for agentic workloads. The result is a model that is simultaneously cheaper to run, better at long context, and more capable on reasoning tasks than any pure Transformer MoE of comparable size.

DeepSeek-V4 (2026) attacks the same O(n²) problem from a different angle: instead of replacing attention entirely, it keeps the Transformer backbone but radically compresses what attention has to look at. Compressed Sparse Attention (CSA) squeezes every m tokens into one KV entry then sparsely selects the top-k relevant blocks; Heavily Compressed Attention (HCA) compresses even more aggressively and does dense attention over the tiny result. Interleaving the two cuts the KV cache by 10× at 1M tokens relative to DeepSeek-V3.2. Two additional innovations compound the gains: Manifold-Constrained Hyper-Connections (mHC) expand the residual stream and constrain the mixing matrix to be doubly stochastic — guaranteeing spectral norm ≤ 1 and numerically stable deep stacking — and the Muon optimizer replaces AdamW for most layers by orthogonalizing gradient updates via Newton-Schulz iterations, yielding faster convergence on ill-conditioned loss landscapes. Post-training follows a specialist-then-distill paradigm: independent expert models are trained per domain via SFT + GRPO, then unified into a single model through on-policy distillation. DeepSeek-V4-Pro-Max sets the open-model SOTA on reasoning and knowledge benchmarks.

The through-line: every paper in this wiki is asking the same question — *how do you get the most intelligence per FLOP?* — and each one pries open a different dimension: computation graph (Transformer), memory management (Mamba), parameter efficiency (MoE), hardware utilization (Nemotron-3), long-context attention efficiency (DeepSeek-V4).

---

## Reading Order

**If you're new:** [[Transformer]] → [[Mixture-of-Experts]] → [[Mamba]] → [[Nemotron-3]]

**If you care about efficiency:** [[Mamba]] → [[Mixture-of-Experts]] → [[Nemotron-3]]

**If you're deploying:** [[Nemotron-3]] (it synthesizes everything; read the others for the why)

**If you want the math:** [[Transformer]] (attention) → [[Mamba]] (SSM discretization + selection) → [[Mixture-of-Experts]] (load balancing loss) → [[Nemotron-3]] (LatentMoE projection + NVFP4 format)
