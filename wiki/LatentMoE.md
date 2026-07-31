---
created: "2026-06-17"
title: "LatentMoE"
authors: "NVIDIA"
year: "2025"
arxiv: ""
technical_report: "https://arxiv.org/pdf/2512.20856"
source_type: "technical_report"
tags: [moe, architecture, routing, efficiency, nvidia]
tldr: "Expert-routing architecture that compresses tokens into a latent space before routing, activating 4x more experts at the same computational cost"
citation_count: 0
---

# LatentMoE

## TL;DR

LatentMoE compresses tokens into a smaller latent dimension before routing them to experts, rather than routing directly from the model's full hidden dimension. This lets a model call on roughly 4× as many expert specialists for the same inference cost. Introduced in NVIDIA's [[Nemotron-3]] (Super and Ultra models) as the core FFN design across all MoE layers.

---

## The Problem

Standard [[Mixture-of-Experts]] routing operates in the full hidden dimension $d$. Two hardware bottlenecks trace directly back to this:

**Throughput (large batch) — all-to-all communication:** When tokens are dispatched to experts on different GPUs, the all-to-all communication volume is proportional to $K \times d$ — active experts × hidden dimension. More GPUs → more experts → more communication. $d$ is the fixed multiplier.

**Latency (small batch) — memory bandwidth:** Reading a single expert's weight matrix from HBM costs bandwidth proportional to $d \times m$ (hidden dim × expert intermediate dim). With $K$ active experts per token, total bandwidth is $K \times d \times m$. Again, $d$ is the multiplier.

As models grow, this routing becomes the bottleneck in both regimes.

---

## The Idea

Project tokens into a smaller latent space first, then route and compute within that compressed representation:

$$x_\ell = W_{\text{down}} \cdot x \quad (d \rightarrow \ell, \text{ where } \ell \ll d)$$

Route and compute expert FFNs in $\ell$-dimensional space. Project back:

$$y = W_{\text{up}} \cdot y_\ell \quad (\ell \rightarrow d)$$

All-to-all communication drops from $K \times d$ to $K \times \ell$. Expert weight bandwidth drops from $d \times m$ to $\ell \times m$.

**The reinvestment:** With the saved budget, scale total experts from $N$ to $N' = N \cdot d/\ell$ and active experts from $K$ to $K' = K \cdot d/\ell$. FFN expressivity (proportional to $K \times m$) now grows by $d/\ell$ at the same hardware cost. The projection matrices $W_{\text{down}}$ and $W_{\text{up}}$ are shared across all experts — minimal parameter overhead.

In [[Nemotron_3_Super|Nemotron 3 Super]]: $d = 4096$, $\ell = 1024$, $d/\ell = 4$. A standard MoE with 128 experts / 6 active becomes **512 experts / 22 active** at the same all-to-all communication volume.

---

## Why It Matters

**Ablation on an 8B active MoE (1T tokens):**

| Model | Experts (total / active) | MMLU-Pro | MMLU | Math | Code |
|---|---|---|---|---|---|
| Standard MoE | 128 / 6 | 48.30 | 70.10 | 78.32 | 51.95 |
| **LatentMoE** | **512 / 22** | **52.87** | **72.11** | **80.19** | **55.14** |

Consistent 4+ point quality improvement across all benchmarks at identical hardware cost.

**It reframes the MoE scaling question.** The standard MoE tradeoff is: more experts → better quality but higher communication cost. LatentMoE breaks that tradeoff — more experts at *lower* communication cost. The new limiting factor is the $W_{\text{down}}$/$W_{\text{up}}$ projections, not the expert GEMMs.

**It fixes MoE's low-latency problem.** Standard MoE is penalized at small batch sizes because memory bandwidth for reading expert weights dominates. Reducing expert matrix size by $d/\ell$ directly reduces this penalty and makes MoE viable in latency-sensitive settings.

**The projection matrices serve double duty.** $W_{\text{down}}$ learns a "routing-friendly" representation of the token — a latent space where expert specialization is maximally informative. This may explain part of the quality gain beyond the pure count increase.

**Scales with model size.** At [[Nemotron_3_Ultra|Nemotron 3 Ultra]] scale (550B total / 55B active), the all-to-all savings are proportionally larger — making LatentMoE more valuable the bigger the model gets.

---

## Related Concepts

*MoE: [[Mixture-of-Experts]] · [[Load Balancing Loss]]*

**Where it appears:** [[Nemotron-3]] (introduced) · **[[Kimi-K3|Kimi K3]]** — "Stable LatentMoE" variant activating 16 of 896 routed experts per token at 2.8T total parameters

*Nemotron 3 family: [[Nemotron-3]] (whitepaper, arXiv 2512.20856) · [[Nemotron_3_Super|Nemotron 3 Super]] · [[Nemotron_3_Ultra|Nemotron 3 Ultra]]*

*Also adopted by: [[MAI-Thinking-1]] (Microsoft AI, cites LatentMoE directly)*

*Co-introduced with: [[NVFP4]] · [[Multi-Token Prediction]] · [[Hardware-Aware Scan]]*
