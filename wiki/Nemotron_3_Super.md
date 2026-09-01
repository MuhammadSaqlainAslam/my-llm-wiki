---
created: "2026-06-17"
title: "Nemotron 3 Super"
authors: "NVIDIA"
year: 2026
arxiv: "2604.12374"
technical_report: "https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf"
source_type: "technical_report"
tags: [model-family, moe, mamba, hybrid-architecture, agentic, nvidia]
tldr: "120B total / 12.7B active hybrid Mamba-2 + sparse attention + LatentMoE model from NVIDIA. 7.5× throughput over Qwen3.5-122B at 1M context. 60.47% on SWE-bench (OpenHands) and 89.0% AIME 2025. Formal arXiv paper: 2604.12374; NVIDIA-hosted technical report also available."
citation_count: 20
---

# Nemotron 3 Super

> NVIDIA, "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning", arXiv:2604.12374, April 2026
> Two sources for this model: the formal arXiv paper ([2604.12374](https://arxiv.org/abs/2604.12374)) and the NVIDIA-hosted technical report ([PDF](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)) — both cover the same model. The Nemotron 3 family overview paper (Nano + Super + Ultra) is a separate document: see [[Nemotron-3]] (arXiv:2512.20856).

Nemotron 3 Super is the mid-tier model in the [[Nemotron-3]] family, positioned at the best accuracy-per-throughput tradeoff of the three variants (Nano, Super, Ultra). The full family architecture is described in the [[Nemotron-3]] whitepaper (arXiv 2512.20856); this note focuses on Super-specific numbers and the dedicated technical report details.

---

## TL;DR

Three simultaneous bets on throughput-first inference:

1. **[[Mamba]]-2 for most layers** — constant-size recurrent state, no KV cache growth with context length
2. **[[LatentMoE]]** — route in a compressed $\ell$-dimensional latent space to 4× more experts at the same communication cost
3. **[[Multi-Token Prediction]]** — auxiliary heads predict multiple future tokens; doubles as free speculative decoding at inference

Result: 7.5× higher throughput than Qwen3.5-122B at 1M-token context, while matching or exceeding it on math, reasoning, and long-context benchmarks.

---

## Architecture

### Dimensions

| Hyperparameter | Value |
|---|---|
| Total parameters | 120B |
| Active parameters / token | 12.7B (incl. embeddings) |
| Layers | 88 |
| Hidden dimension | 4096 |
| Q-heads / KV-heads | 32 / 2 (GQA) |
| Mamba-2 state dimension | 128 |
| Total experts | 512 |
| Active experts / token | 22 |
| MoE latent dimension ($\ell$) | 1024 |
| MTP layers | 2 (shared weights) |

### Layer Pattern

Each block repeats:
```
[Mamba-2, LatentMoE, Mamba-2, LatentMoE, Mamba-2, Attention, LatentMoE]
```

The attention layers are sparse and strategic — they handle exact recall for tasks where Mamba's selective compression loses information (e.g., copying a token verbatim from early context). Everything between them runs as Mamba-2, which has no KV cache.

### LatentMoE

Standard MoE all-to-all communication scales as $K \times d$ (active experts × hidden dim). LatentMoE projects tokens $d \rightarrow \ell$ before routing, cutting this to $K \times \ell$.

With $d = 4096$ and $\ell = 1024$ (4× compression), reinvest the saved bandwidth: increase from 128 total / 6 active experts to **512 total / 22 active** at the same communication cost. More experts = more specialized FFN capacity at no extra inference overhead.

Ablation on an 8B active MoE (1T tokens):

| | Standard MoE | LatentMoE |
|---|---|---|
| Experts (total / active) | 128 / 6 | 512 / 22 |
| MMLU-Pro | 48.30 | **52.87** |
| MATH | 78.32 | **80.19** |
| Code | 51.95 | **55.14** |

### Multi-Token Prediction (MTP)

Two MTP layers with shared weights predict 2 and 3 tokens ahead simultaneously. Payoff is double:
- **Training**: richer gradient signal per token — the model must plan ahead
- **Inference**: MTP predictions become speculative drafts at ~97% acceptance rate — effectively free tokens per forward pass

### NVFP4 Training

Most linear layers trained in NVFP4 (E2M1, 16-element micro-block scaling). Sensitive layers (QKV, latent projections, final 15% of layers) stay in BF16. Mamba output projections use MXFP8. Result: < 0.6% loss gap vs. BF16, 3× hardware throughput on GB300 / B200.

### 1M Token Context

Mamba layers carry a constant-size recurrent state regardless of sequence length — no KV cache growth. The few attention layers use GQA (2 KV heads) keeping their cache tiny. No RoPE (which degrades when extrapolating beyond training length); positional information comes from Mamba's recurrent dynamics.

Long-context training: 34B tokens at 1M sequence length, plus 17B alternating 1M/4K tokens, then SFT at 256K and RL at 32K.

### Training Data

25 trillion tokens. Includes 15M synthetic coding problems, algorithmic reasoning, economics, formal logic, and MCQ data. Checkpoint merging (minus-sqrt weighting over 125B/250B/500B windows) adds 2–4 benchmark points for free at the same compute budget.

---

## Post-Training Pipeline

Four sequential stages:

```
SFT (7M samples, 800B tokens)
 → [[Multi-Environment RLVR Training|RLVR]] (21 environments, simultaneous, async GRPO)
 → SWE-RL (software engineering, OpenHands scaffold)
 → RLHF (reward model, helpfulness/harmlessness)
 → MTP Healing (re-align speculative heads to post-trained distribution)
```

SFT blend: 36% agentic, 31% reasoning, 23% chat, 8% long context. RLVR trains all 21 environments simultaneously (not sequentially) for more stable convergence and less capability regression. GRPO: 256 prompts × 16 responses, 64K max generation length.

---

## Why It Matters

### Throughput

At 8K input / 64K output on 8× B200 GPUs vs. comparable models:

| Model | Relative throughput |
|---|---|
| **Nemotron-3-Super-120B** | 1× (baseline) |
| GPT-OSS-120B (Transformer MoE) | 0.45× |
| Qwen3.5-122B (Transformer MoE) | 0.13× |

The 7.5× advantage over Qwen3.5-122B is almost entirely due to Mamba's constant-state recurrence — the gap widens further at longer output sequences.

### Accuracy (post-trained)

| Benchmark | Nemotron-3-Super | Qwen3.5-122B | GPT-OSS-120B |
|---|---|---|---|
| HMMT Feb 2025 | **94.73** | 89.55 | 91.20 |
| RULER @ 1M | **91.64** | 91.33 | 84.50 |
| SWE-Bench (OpenHands) | 60.47 | **66.40** | 61.20 |
| AIME 2025 | **89.0** | 85.3 | 86.5 |
| Arena-Hard (with tools) | **99.2** | 98.7 | 98.1 |

SWE-Bench is the exception where Qwen3.5 leads. Every other category Super is competitive or first.

Reasoning efficiency (MTP SPEED-Bench — accuracy per output token): Super scores **3.45** vs. competitors around 2.65–2.80. The speculative drafts from MTP heads directly improve this metric.

---

## Related Concepts

*Family: [[Nemotron-3]] (whitepaper, arXiv 2512.20856) · [[Nemotron_3_Ultra]] (550B/55B active)*

*Architecture: [[Mamba]] · [[Mixture-of-Experts]] · [[LatentMoE]] · [[Multi-Token Prediction]] · [[GQA]] · [[NVFP4]]*

*Training: [[RLVR]] · [[Multi-Environment RLVR Training]] · [[GRPO]] · [[Hardware-Aware Scan]]*
