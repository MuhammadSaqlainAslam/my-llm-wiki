---
title: "Nemotron 3 Ultra"
authors: "NVIDIA"
year: 2026
arxiv: ""
technical_report: "https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf"
source_type: "technical_report"
tags: [model-family, moe, mamba, hybrid-architecture, agentic, nvidia]
tldr: "550B total / 55B active hybrid Mamba-Attention MoE, the largest and most capable model in the Nemotron 3 family. Released June 4 2026. Trained with Multi-Teacher On-Policy Distillation (MOPD) from 10+ specialist teacher models. Artificial Analysis quality score 48. No arXiv submission — technical report published by NVIDIA Research."
citation_count: 0
---

# Nemotron 3 Ultra

> NVIDIA, "Nemotron 3 Ultra Technical Report", 2026
> Official technical report: [NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) via NVIDIA Research — no arXiv submission exists for this model.

Nemotron 3 Ultra is the largest model in the [[Nemotron-3]] family, released June 4 2026. It uses the same hybrid Mamba-Transformer MoE backbone as [[Nemotron_3_Super]] but at 4.5× the scale, and adds a new post-training technique — Multi-Teacher On-Policy Distillation (MOPD) — to transfer knowledge from more than 10 specialist teacher models into one unified set of weights.

---

## TL;DR

Ultra is what happens when you apply the [[Nemotron-3]] architectural playbook at maximum scale and invest heavily in the post-training stage. The architectural bets are the same as Super (Mamba recurrence, LatentMoE, MTP); the differentiator is MOPD — a distillation method that extracts diverse capabilities from a fleet of specialist teachers simultaneously, avoiding the quality loss from naive multi-task RL.

At quality score 48 on the Artificial Analysis benchmark, Ultra is competitive with the leading frontier closed models while remaining open-weight under Apache 2.0.

---

## Architecture

### Dimensions

| Hyperparameter | Value |
|---|---|
| Total parameters | 550B |
| Active parameters / token | 55B |
| Backbone | Hybrid Mamba-2 + sparse attention + LatentMoE |
| MTP layers | Yes (speculative decoding at inference) |
| Context length | 1M tokens |
| Training precision | NVFP4 (mixed precision) |

The layer pattern mirrors Super but at greater depth and hidden dimension. [[LatentMoE]] expands expert count without increasing all-to-all communication — at Ultra's scale the communication savings are even more significant than at Super scale. Mamba-2 layers carry a constant recurrent state so the KV cache remains bounded regardless of the 1M-token context window.

---

## Post-Training: Multi-Teacher On-Policy Distillation (MOPD)

The key innovation distinguishing Ultra from a simple scale-up of Super.

### The problem with multi-task RL

Training one model to be best at coding, math, reasoning, instruction following, and agentic tasks simultaneously via RL leads to capability conflict — improving one task's reward often regresses another. The typical solution (sequential per-capability RL) causes its own problems: the final stage overwrites earlier gains.

### MOPD

**Step 1 — Grow specialists.** Train 10+ separate teacher models, each expert in one domain (competitive math, competitive coding, SWE-bench, instruction following, long-context, etc.). Each teacher undergoes targeted SFT + RL in its domain without competing with other domains.

**Step 2 — On-policy distillation.** The student (Ultra) generates its own rollouts (on-policy). For each rollout, compute the reverse KL divergence against all relevant teacher specialists and minimize the aggregated divergence. The student learns to simultaneously match the behavior of every specialist.

Why reverse KL: it mode-seeks, meaning the student strongly avoids outputting things no teacher would say. This produces a conservative, high-precision behavior policy rather than a broad low-confidence one — desirable for a model expected to be reliable.

**Multi-teacher aggregation:** Different teachers are relevant for different prompts. A coding prompt weights the coding teacher heavily; a math proof weights the math teacher. The aggregation is prompt-dependent, determined by domain classifiers on the rollout prompt.

**Result:** One set of weights captures capabilities that previously required separate deployed models — without the quality degradation of naive model merging (model soups).

---

## Why It Matters

### Benchmark Performance

Artificial Analysis quality score: **48** — competitive with the leading frontier closed models as of June 2026.

Throughput comparison at long output lengths vs. comparable dense-equivalent size models:

| Model | Relative throughput |
|---|---|
| **Nemotron-3-Ultra-550B** | 1× (baseline) |
| GLM comparable size | lower (Transformer MoE) |
| Kimi comparable size | lower (Transformer MoE) |
| Qwen comparable size | lower (Transformer MoE) |

The Mamba recurrence advantage compounds at Ultra scale: more layers means the relative advantage of O(1)-per-step inference vs. growing KV caches is larger.

### Open Weights

Released under Apache 2.0. Full post-training stack (NeMo-RL, NeMo-Gym), training recipes, and data (where redistribution rights permit) are released. This makes Ultra the largest open-weight hybrid Mamba-Transformer model available as of June 2026.

---

## Relationship to the Family

| Model | Params (total / active) | Key differentiator |
|---|---|---|
| Nano | 30B / 3B | Best throughput; 3.3× faster than Qwen3-30B-A3B |
| [[Nemotron_3_Super\|Super]] | 120B / 12.7B | Best accuracy-throughput balance; 7.5× faster than Qwen3.5-122B |
| **Ultra** | 550B / 55B | Highest capability; MOPD from 10+ teacher specialists |

All three use the same architectural backbone described in the [[Nemotron-3]] whitepaper (arXiv 2512.20856). Ultra adds MOPD as the critical post-training differentiator.

---

## Related Concepts

*Family: [[Nemotron-3]] (whitepaper, arXiv 2512.20856) · [[Nemotron_3_Super]] (120B/12.7B active)*

*Architecture: [[Mamba]] · [[Mixture-of-Experts]] · [[LatentMoE]] · [[Multi-Token Prediction]] · [[GQA]] · [[NVFP4]]*

*Training: [[RLVR]] · [[On-Policy Distillation]] · [[GRPO]]*
