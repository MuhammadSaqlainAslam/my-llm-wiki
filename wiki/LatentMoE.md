---
title: "LatentMoE"
tags: [moe, routing, efficiency, nemotron, communication]
tldr: "Project tokens from d to a smaller latent dimension ℓ before MoE routing. All-to-all communication drops by d/ℓ; reinvest the savings into more experts. Same hardware cost, 4+ point quality improvement."
theme: scaling
---

# LatentMoE

Standard [[Mixture-of-Experts]] routing operates in the full hidden dimension $d$. Two hardware bottlenecks trace directly back to this choice:

**Multi-GPU communication (throughput setting):** When tokens are dispatched to experts on different GPUs, the all-to-all communication volume is proportional to $K \times d$ — the number of active experts times the hidden dimension. More GPUs = more experts = more communication. $d$ is the fixed multiplier you can't escape.

**Memory bandwidth (latency setting):** Reading a single expert's weight matrix from GPU HBM costs bandwidth proportional to $d \times m$, where $m$ is the expert's intermediate dimension. With $K$ active experts per token, total bandwidth is $K \times d \times m$. Again, $d$ is the multiplier.

**The fix:** project tokens into a smaller latent space before routing.

$$x_{\ell} = W_{\text{down}} \cdot x \quad (d \rightarrow \ell, \text{ where } \ell < d)$$

Route and compute expert FFNs in $\ell$-dimensional space. Project back:

$$y = W_{\text{up}} \cdot y_{\ell} \quad (\ell \rightarrow d)$$

Communication is now $K \times \ell$ instead of $K \times d$. Expert weight bandwidth is $\ell \times m$ instead of $d \times m$. If $d/\ell = 4$, both bottlenecks shrink by 4×. Reinvest the savings: scale total experts from $N$ to $N \cdot (d/\ell)$ and active experts from $K$ to $K \cdot (d/\ell)$. The FFN expressivity (proportional to $K \times m$) grows by $d/\ell$ at the same hardware cost. The projection matrices $W_{\text{down}}$ and $W_{\text{up}}$ are shared across all experts, so they add minimal parameter overhead.

**Nemotron-3 Super numbers:** $d = 4096$, $\ell = 1024$, $d/\ell = 4$. A baseline of 128 experts / 6 active in standard MoE becomes 512 experts / 22 active in LatentMoE with identical all-to-all communication volume.

**Ablation (8B active / ~73B total, 1T tokens):**

| Model | Experts (total / active) | MMLU-Pro | MMLU | Math | Code |
|---|---|---|---|---|---|
| Standard MoE | 128 / 6 | 48.30 | 70.10 | 78.32 | 51.95 |
| **LatentMoE** | **512 / 22** | **52.87** | **72.11** | **80.19** | **55.14** |

## Where it appears

- **[[Nemotron-3]]** — the core FFN design in both Nano and Super; every MoE layer in the network is a LatentMoE layer
- **[[Mixture-of-Experts]]** — mentioned briefly as the solution to MoE's communication bottleneck at scale

## Why it matters

- **It reframes the MoE scaling question.** The usual MoE tradeoff is: more experts → better quality but higher communication cost. LatentMoE breaks that tradeoff — more experts at *lower* communication cost. The limiting factor becomes the $W_{\text{down}}$/$W_{\text{up}}$ projections, not the expert GEMMs.
- **It's a practical answer to the "MoE doesn't work at low latency" critique.** Standard MoE is penalized at small batch sizes because memory bandwidth for reading expert weights dominates. Reducing expert matrix size by $d/\ell$ directly reduces this penalty and makes MoE viable in lower-throughput settings.
- **The projection matrices serve double duty.** $W_{\text{down}}$ can be interpreted as learning a "routing-friendly" representation of the token — the latent space that makes expert specialization maximally informative. This may explain part of the quality gain over standard MoE beyond the pure count increase.

---

*Related: [[Mixture-of-Experts]] · [[Load Balancing Loss]] · [[Nemotron-3]]*
