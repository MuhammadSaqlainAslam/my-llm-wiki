---
created: "2026-07-30"
title: "Kimi K3: Open Frontier Intelligence"
authors: "Kimi Team, Moonshot AI"
year: 2026
tags: [model-family, moe, agentic, reasoning, moonshot, vision, latent-moe]
aliases: [Kimi K3, K3]
tldr: "2.8T-parameter MoE (104B activated) built on Kimi Delta Attention and Attention Residuals, with Stable LatentMoE activating 16 of 896 routed experts per token. ~2.5x more scaling-efficient than Kimi K2; open-weight, native vision, 1M-token context, and consistently outperforms other open and proprietary models while still trailing Claude Fable 5 and GPT-5.6 Sol."
theme: synthesis
arxiv: "2607.24653"
citation_count: 0
---

# Kimi K3: Open Frontier Intelligence

## TL;DR

Kimi K3 is Moonshot AI's follow-up to [[Kimi-K2|Kimi K2]]: a 2.8T-parameter [[Mixture-of-Experts]] model with 104B activated parameters, native vision capabilities, and a 1-million-token context window. It combines three architectural pieces — **Kimi Delta Attention**, **[[Attention Residuals]]**, and **Stable LatentMoE** — to get roughly 2.5x better overall scaling efficiency than Kimi K2. It's open-weight, and while it still trails the strongest proprietary models (Claude Fable 5, GPT-5.6 Sol), it consistently outperforms other open and proprietary models across the evaluated suite.

## Architecture

- **Kimi Delta Attention** and **[[Attention Residuals]]** — together improve information flow, respectively across sequence length and across model depth. Attention Residuals replaces the fixed-weight PreNorm residual sum with learned, input-dependent attention over preceding layers (see that note for detail); Kimi Delta Attention is the sequence-length-side counterpart.
- **Stable LatentMoE** — a refined version of the latent-space expert routing introduced by [[LatentMoE|Nemotron-3's LatentMoE]]: tokens are projected into a latent space before routing, and only 16 of 896 routed experts activate per token, keeping per-token compute low despite the very high total expert count.
- **Scale**: 2.8T total parameters, 104B active per token, 1M-token context, native vision input.

## Training

Post-training emphasizes reinforcement learning across general, agentic, and coding domains, at multiple reasoning-effort levels — aimed at compositional generalization and robust long-horizon execution rather than single-turn benchmark performance alone. At this scale, the paper also describes infrastructure work required to make training practical: algorithm-system co-design for Kimi Delta Attention, balanced expert-parallel training with efficient memory management, and million-token agentic RL with persistent rollout and sandbox state.

## Key Results

- ~2.5x improvement in overall scaling efficiency versus Kimi K2.
- Frontier-level performance across long-horizon coding, agentic, knowledge, reasoning, and vision tasks.
- Still trails the strongest proprietary models (Claude Fable 5, GPT-5.6 Sol) overall, but consistently outperforms other open and proprietary models evaluated in the same suite.
- Full model weights released.

## Why It Matters

K3 is a concrete demonstration that the architectural ideas introduced separately — [[Attention Residuals]] (depth-wise information flow) and latent-space MoE routing (introduced by Nemotron-3, refined here) — compose well together at frontier scale, and that doing so buys real scaling efficiency rather than just marginal benchmark gains. Releasing full weights at 2.8T/104B-active scale also keeps the open-weight frontier close behind the best closed models.

## Limitations

- Still behind the top proprietary models overall, despite leading other open/proprietary comparisons.
- 104B active parameters and 2.8T total is still a very large deployment footprint even with sparse activation.

## Related Concepts

[[Kimi-K2|Kimi K2]] · [[Attention Residuals]] · [[LatentMoE]] · [[Mixture-of-Experts]] · [[Nemotron-3]]
