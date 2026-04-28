---
title: "Nemotron 3: Efficient and Open Intelligence"
authors: "NVIDIA"
year: 2025
tags: [hybrid, moe, mamba, production, inference, quantization, rl, agentic]
tldr: "Hybrid Mamba-2 + sparse attention + LatentMoE, trained with NVFP4 precision and multi-token prediction. 120B total / 12B active. 7.5× throughput over Qwen3.5-122B at 1M context, with competitive accuracy across math, code, and agentic tasks."
theme: synthesis
---

# Nemotron 3

> NVIDIA, "Nemotron 3: Efficient and Open Intelligence", December 2025  
> NVIDIA, "Nemotron 3 Super Technical Report", 2025

## What It Is

Nemotron 3 is a family of three models — Nano (30B total / 3B active), Super (120B total / 12B active), and Ultra — built around one core constraint: **inference throughput matters as much as quality**. The architecture bets on hybrid Mamba-Transformer MoE, where most layers are cheap [[Mamba]] SSMs and a minority are attention layers providing exact recall, all mixed with [[Mixture-of-Experts]] FFN layers.

The three technologies together:
1. **Hybrid Mamba-Transformer MoE** — mostly [[Mamba]]-2 layers, sparse attention, LatentMoE FFNs
2. **LatentMoE** — route tokens in a compressed latent space to cut communication cost and expand expert count
3. **Multi-Token Prediction (MTP)** — auxiliary heads predict multiple future tokens; used for training signal and free speculative decoding at inference

---

## Architecture: Nemotron 3 Super

**Model dimensions:**

| Hyperparameter | Value |
|---|---|
| Total parameters | 120B |
| Active parameters / token | 12B (12.7B including embeddings) |
| Layers | 88 |
| Hidden dimension ($d_{\text{model}}$) | 4096 |
| Q-heads / KV-heads | 32 / 2 (GQA) |
| Mamba-2 state dimension | 128 |
| Total experts | 512 |
| Active experts / token | 22 |
| MoE latent dimension | 1024 |
| MTP layers | 2 (shared weights) |

**Layer pattern** (repeated block):
```
[Mamba-2, LatentMoE, Mamba-2, LatentMoE, Mamba-2, Attention, LatentMoE] × N
```

The attention layers are sparse and strategic — they provide the exact recall that Mamba struggles with. Mamba handles cheap sequence processing everywhere else. With only a few attention layers, the KV cache stays tiny even at long context.

**Nano layer pattern** (for reference):
```
[Mamba-2, MoE, Mamba-2, MoE, Mamba-2, Attention, MoE] × 5
[Mamba-2, MoE, Mamba-2, MoE, Mamba-2, Attention, MoE] × 3
[Mamba-2, MoE, Mamba-2, Attention, MoE]              × 1
[Mamba-2, MoE, Mamba-2, MoE, Mamba-2, MoE, Mamba-2, MoE] × 4
```

---

## LatentMoE: Hardware-Aware Expert Design

Standard MoE has two deployment bottlenecks:

**Latency-focused (small batch):** Memory bandwidth dominates. Reading expert weight matrices from memory costs more than the computation itself. Expert matrix size = $d \times m$ where $d$ is hidden dimension and $m$ is expert intermediate dimension.

**Throughput-focused (large batch):** All-to-all communication dominates. Dispatching tokens to experts scales with $K \times d$ (active experts × hidden dimension) — independent of $m$.

Notice: the bottleneck in both cases is $d$. The fix: **route in a smaller latent space**.

**LatentMoE:**
1. Project each token: $d \rightarrow \ell$ (latent dimension, typically $\ell = d/4$)
2. Route and compute in $\ell$-dimensional space
3. Project back: $\ell \rightarrow d$

All-to-all communication drops from $K \times d$ to $K \times \ell$. Memory bandwidth for expert weights drops from $d \times m$ to $\ell \times m$.

With the saved budget, reinvest: increase total experts from $N$ to $N' = N \cdot d/\ell$, increase active experts from $K$ to $K' = K \cdot d/\ell$. The FFN expressivity (proportional to $K \times m$) is now $K' \times m$ — larger, better quality.

In Super: $d = 4096$, $\ell = 1024$, so $d/\ell = 4$. A standard MoE with 128 experts / 6 active becomes 512 experts / 22 active in LatentMoE at the same communication cost.

**Ablation results (8B active / ~73B total, 1T tokens):**

| Model | Experts (total / active) | MMLU-Pro | MMLU | Math | Code |
|---|---|---|---|---|---|
| Standard MoE | 128 / 6 | 48.30 | 70.10 | 78.32 | 51.95 |
| **LatentMoE** | **512 / 22** | **52.87** | **72.11** | **80.19** | **55.14** |

LatentMoE consistently wins at essentially the same inference cost.

---

## Multi-Token Prediction (MTP)

Standard LM training: predict next token, compute loss, update. One gradient signal per position.

MTP adds auxiliary heads that predict 2, 3, ... tokens ahead simultaneously. Benefits:
1. **Richer training signal**: the model must plan ahead, not just predict next token
2. **Better reasoning**: anticipating future tokens encourages structured internal representations
3. **Free speculative decoding**: auxiliary predictions become draft tokens — ~97% acceptance rate on first two drafts in ablation

Super uses 2 MTP layers with **shared weights** — same parameters as the base model layers, reducing extra parameter count. The MTP heads are used both during training and during RL rollout to speed up token generation.

On an 8B active MoE base model (1T tokens), MTP improves **~2.4% average across benchmarks**:

| Task | Baseline | +MTP |
|---|---|---|
| MMLU (5-shot) | 70.06 | **71.26** |
| MMLU-Pro (CoT) | 45.05 | **47.84** |
| MBPP-Sanitized | 65.58 | **66.89** |
| ARC-Challenge | 86.43 | **88.05** |
| GSM8K (8-shot) | 82.49 | **84.46** |

---

## NVFP4 Training

NVIDIA's proprietary 4-bit floating point format for training. Enables 3x higher peak throughput on GB300 / B200 hardware versus BF16.

**NVFP4 format:**
- Element format: E2M1 (2 exponent bits, 1 mantissa bit)
- 16-element micro-block scaling (fine-grained, not per-tensor)
- FP32 global scale + E2M1 block scaling factors
- 2D block scaling for weight quantization
- Random Hadamard Transforms on weight gradient inputs (RHTs)
- Stochastic rounding on gradients

**Precision recipe by layer type (Super):**

| Layer / Operation | Precision |
|---|---|
| Most linear layers (dense GEMMs) | NVFP4 |
| Final 15% of network layers | BF16 |
| Latent projections (LatentMoE) | BF16 |
| MTP layers | BF16 |
| QKV / attention projections | BF16 |
| Mamba output projections | MXFP8 |
| Embeddings | BF16 |

Mamba output projections and QKV/attention projections are numerically sensitive — they can flush to zero (up to 40% on Nano) when quantized to NVFP4. Keeping them in higher precision eliminates this.

**Results with correct mixed-precision recipe:**
- Nano: < 1% relative difference in loss vs. BF16
- 8B active MoE model: < 0.6% loss gap
- Downstream task accuracy matches BF16 models
- Loss gap shrinks with model size (larger models are more quantization-robust)

---

## Pretraining

**Data:** 25 trillion tokens

**Optimizer:** AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$

**Sequence length:** 8192 tokens

**Batch size:** 3072 sequences

**Learning rate schedule:** 2-phase WSD (Warmup-Stable-Decay):
- Warmup: 0 → 200B tokens, LR ramps to $4.5 \times 10^{-4}$
- Stable plateau: 200B → ~20T tokens, constant LR
- Decay: final 5T tokens, minus-sqrt decay

**Routing:** Auxiliary-loss-free load balancing (bias update rate $10^{-3}$) + standard auxiliary loss (coefficient $10^{-4}$), sigmoid router (not softmax).

**Synthetic data in pretraining corpus:**
- 15M synthetic coding problems (structured algorithmic challenges)
- Algorithmic reasoning data
- Economics and formal logic reasoning
- Multiple-choice questions (MCQ) at scale

---

## Checkpoint Merging

During the stable training phase, checkpoints from different training windows are merged by taking a weighted combination of model weights. The merge coefficients follow minus-sqrt weighting (same as the LR decay shape).

Windows used: 125B token windows, 250B windows, 500B windows.

**Effect:** 2–4 point improvement in benchmark scores compared to the final un-merged checkpoint at the same compute budget. This is essentially free quality — no extra training cost.

The intuition: different checkpoints specialize in different local optima. Merging averages away overfitting artifacts and improves generalization.

---

## Long Context: Up to 1M Tokens

**Why Mamba helps here:** Mamba layers have a constant-size recurrent state regardless of sequence length. No KV cache grows with context. The few attention layers in Super use GQA (2 KV heads), keeping their cache small.

**No RoPE:** Rotary position embeddings degrade when extrapolating beyond training length. Mamba captures positional information through its recurrent dynamics. The attention layers also skip RoPE.

**LC-Phase training:**
1. **Continued Pre-Training (CPT)** at 1M sequence length: 34B tokens
2. **Alternating** 1M-context and 4K-context data: 17B additional tokens
3. SFT at 256k context, RL at 32k context (with long-context synthetic rollouts)

**Context length improvement over previous generation:**

| Model | RULER score @ 1M tokens |
|---|---|
| Nemotron-Nano-12B-v2-Base (dense hybrid) | 23.43 |
| **Nemotron-3-Nano-30B-A3B-Base (MoE hybrid)** | **54.19** |
| **Nemotron-3-Super-120B-Base** | **71.00** |

The MoE hybrid degrades gracefully at 1M; the dense hybrid falls off a cliff between 512k and 1M.

NLL on code decreases monotonically up to 1M tokens — the model genuinely uses long context rather than ignoring it.

---

## Base Model Results

**Nemotron-3-Super-120B-Base** vs. comparable models:

| Benchmark | Nemotron-3-Super | Ling-flash | GLM-4.5-Air |
|---|---|---|---|
| MMLU (5-shot) | **86.01** | 85.00 | 84.95 |
| MMLU-Pro (CoT) | **75.65** | 73.55 | 72.40 |
| MATH (CoT) | **84.84** | 82.11 | 81.03 |
| AIME 2024 (pass@32) | **53.33** | 49.44 | 45.56 |
| HumanEval (0-shot) | **79.40** | 76.83 | 75.00 |
| RULER @ 1M | **71.00** | 60.12 | 58.30 |

Base model beats comparable-size alternatives across all categories before any post-training.

---

## Post-Training Pipeline

Four sequential stages, each building on the previous:

```
SFT → RLVR (21 environments) → SWE-RL → RLHF → MTP healing
```

### Stage 1: Supervised Fine-Tuning

Two-sub-stage SFT:
1. **Token-level loss** (standard cross-entropy on every token)
2. **Sample-level loss** (loss weighted by sample quality score)

**SFT data blend (7M samples, ~800B tokens):**

| Category | Fraction |
|---|---|
| Agentic (tool use, multi-turn) | 36% |
| Reasoning (math, code, logic) | 31% |
| Chat / instruction following | 23% |
| Long context | 8% |
| Other | 2% |

Heavy investment in agentic data reflects the design goal: Nemotron 3 is built for agentic workloads, not just chat.

### Stage 2: RLVR (21 Environments)

All environments trained **simultaneously in a single RL run** (unlike prior NVIDIA models that did sequential per-capability RL).

**Environments include:** competitive math, competitive coding, software engineering, instruction following, search, chat, agentic tool use, long context, economics, formal logic, MCQ, and more (21 total across 37 datasets).

Why simultaneous training:
- More stable (less reward hacking)
- Less capability degradation from earlier stages
- Better cross-task generalization

**Implementation:** Async GRPO (Group Relative Policy Optimization)
- 256 prompts per step
- 16 responses per prompt
- Effective batch size: 4096
- Max generation length: 64k tokens
- MTP tokens used as free speculative decoding drafts during rollout — critical for throughput when sampling thousands of 64K-token reasoning traces

**Masked importance sampling:** Handles distribution mismatch between training policy and rollout policy.

### PivotRL

A sample-efficiency technique used during RLVR. Instead of regenerating all responses from scratch each step, PivotRL reuses **offline SFT traces** and focuses RL training on "pivot" turns — the turns where the model has high uncertainty or where the conversation branches.

Effect: more efficient use of rollout budget, faster convergence on hard problems.

### Stage 3: SWE-RL

Dedicated reinforcement learning stage for software engineering:
- Environments: SWE-bench tasks (real GitHub issues requiring code edits)
- Reward: whether the submitted patch passes the test suite
- Uses the OpenHands scaffold for code execution

Trained after RLVR to specialize the model on interactive coding tasks without disrupting the general capabilities built in stage 2.

### Stage 4: RLHF

Standard RLHF stage using a trained reward model. Polishes helpfulness, harmlessness, and honesty for general chat and instruction following. Fine-tunes after all the capability training to ensure good user experience.

### MTP Healing

After post-training changes the distribution of the base model, the MTP heads (which were trained on base model outputs) need to be re-aligned. A short fine-tuning pass re-trains the MTP heads on post-trained model outputs. This restores the ~97% speculative decoding acceptance rate.

---

## Inference Quantization

### FP8 PTQ (Post-Training Quantization)

Baseline quantization for deployment:

| Component | Precision |
|---|---|
| MoE GEMMs | FP8 (W8A8) |
| KV cache | FP8 |
| Mamba GEMMs | FP8 |
| SSM state cache | FP16 |

### NVFP4 PTQ

More aggressive quantization using **AutoQuantize** — an NAS-style search over which layers can be safely quantized to NVFP4 vs. must stay in higher precision.

Process: evaluate each layer's quantization sensitivity, find the Pareto-optimal precision assignment for target throughput.

- **Time:** < 2 hours on 8× B200 GPUs
- **Accuracy:** 99.8% median accuracy vs. BF16 across benchmarks

What gets NVFP4: sparse expert GEMMs (the dominant compute)  
What stays in BF16: attention projections, latent projections, sensitive layers

### Mamba SSM Cache Quantization

The SSM recurrent state (shape: `[batch, layers, state_dim, d_model]`) is a significant memory consumer at large batch sizes.

Quantizing this cache to INT8 causes **verbosity increase** — the model starts generating longer, repetitive text. The culprit is outliers in the SSM state corrupting the recurrence.

Fix: **FP16 with stochastic rounding** using Philox RNG (a hardware-efficient counter-based PRNG). This avoids the verbosity issue while still reducing memory vs. FP32.

Specifically: `Philox<5>` generator, stochastic rounding on quantize, deterministic dequantize.

---

## Throughput vs. Competitors

At 8k input / 64k output, on 8× B200 GPUs:

| Model | Throughput (tokens/s) | Relative |
|---|---|---|
| **Nemotron-3-Super-120B** | — | **1× (baseline)** |
| GPT-OSS-120B (Transformer MoE) | — | 0.45× (2.2x slower) |
| Qwen3.5-122B (Transformer MoE) | — | 0.13× (7.5x slower) |

The hybrid Mamba architecture's throughput advantage grows with output length. At short sequences the gap is smaller; at long outputs the constant-state Mamba layers dominate.

**Nano throughput:** Nemotron-3-Nano-30B achieves **3.3x higher throughput** than Qwen3-30B-A3B at 8k input / 16k output.

---

## Post-Trained Benchmark Results

**Nemotron-3-Super-120B (post-trained) vs. comparable models:**

| Benchmark | Nemotron-3-Super | Qwen3.5-122B | GPT-OSS-120B |
|---|---|---|---|
| HMMT Feb 2025 | **94.73** | 89.55 | 91.20 |
| RULER @ 1M | **91.64** | 91.33 | 84.50 |
| SWE-Bench (OpenHands) | 60.47 | **66.40** | 61.20 |
| AIME 2025 | **89.0** | 85.3 | 86.5 |
| Arena-Hard (with tools) | **99.2** | 98.7 | 98.1 |

SWE-Bench is the exception — Qwen3.5 leads there. Everywhere else, Super is competitive or leading.

**MTP SPEED-Bench** (measures reasoning efficiency — accuracy per token, not just accuracy):

| Model | Average score |
|---|---|
| **Nemotron-3-Super** | **3.45** |
| Competitor A | 2.80 |
| Competitor B | 2.65 |

MTP speculative decoding contributes here: generating more tokens per wall-clock second improves the accuracy-per-second metric.

---

## Reasoning Budget Control

At inference time, users can specify a maximum number of "thinking" tokens. When the model reaches the budget:
- Append `</think>` to the partial thinking trace
- Let the model generate the final answer from the partial trace

This gives a smooth accuracy-efficiency tradeoff curve. On AIME25: performance scales roughly linearly with token budget from 2K to 32K tokens (~20% to ~85% accuracy).

Useful for latency-sensitive deployments: pay more tokens for harder problems, fewer for simpler ones.

---

## Model Family

| Model | Size | Description |
|---|---|---|
| **Nano** | 30B total, 3B active | Best throughput. Released with white paper + technical report. |
| **Super** | 120B total, 12B active | Best accuracy/throughput balance. Technical report released. |
| **Ultra** | TBD | Largest. State-of-the-art accuracy. Full post-training stack. |

All weights, pre/post-training software (NeMo-RL, NeMo-Gym), training recipes, and data (where redistribution rights permit) released openly under Apache 2.0.

---

## Key Takeaway

Nemotron 3 Super makes four architectural bets simultaneously: Mamba for cheap sequence processing, sparse attention for exact recall, LatentMoE for parameter efficiency without communication overhead, and MTP for training signal density plus free speculative decoding. Every choice traces back to "make generation faster without hurting quality."

The result: 7.5x higher throughput than Qwen3.5-122B at comparable accuracy, in a model that also supports 1M-token context and genuine agentic capability from 21-environment RL training.

The other lesson: quantization is not afterthought. NVFP4 training from scratch (not PTQ) and careful per-layer precision assignment are first-class engineering decisions that determine what's achievable on real hardware.

---

*Related: [[Mamba]] · [[Mixture-of-Experts]] · [[Transformer]]*
