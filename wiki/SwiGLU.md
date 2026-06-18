---
created: "2026-06-03"
title: "SwiGLU"
tags: [glossary, activation-functions]
tldr: "A gated activation function (Swish × linear gate) used in the FFN layers of most modern LLMs, empirically outperforming ReLU/GeLU by ~1% perplexity."
---

## TL;DR
SwiGLU is a gated variant of the Swish activation applied in feed-forward layers: `FFN(x) = (xW₁ ⊙ Swish(xV)) W₂`, requiring 3 weight matrices instead of 2 but consistently beating ReLU and GeLU on language modeling benchmarks.

## Intuition
Standard FFN layers apply a non-linearity blindly — every neuron fires through the same gate. SwiGLU adds a learned multiplicative gate: one linear projection decides *what to keep*, another decides *how much*, and Swish (x·σ(βx)) provides smooth, non-zero gradients for negative inputs unlike ReLU. The result is a soft, data-dependent filter on the hidden dimension.

To keep parameter count neutral vs. the original 2-matrix FFN, the hidden dim is typically scaled from 4× to ~2.67× (e.g., LLaMA-2 7B uses hidden=4096, FFN intermediate=11008 ≈ 2.67×). PaLM, LLaMA, Mistral, and Gemini all use SwiGLU — it's become the de facto FFN activation for frontier models after Noam Shazeer's 2020 paper showed consistent ~1 nats/token improvement across model sizes.

## Why It Matters
- **Empirical wins compound at scale**: SwiGLU's perplexity gains over GeLU hold from 125M to 540B parameters, making it a low-risk, high-reward architectural choice baked into most modern LLM recipes.
- **FFN dominates compute**: FFN layers consume ~2/3 of a Transformer's FLOPs; a better activation here multiplies across every layer and every token — small gains matter enormously at inference scale.
- **Interacts with MoE routing**: In sparse MoE models, SwiGLU experts maintain the gating intuition at two levels — the router selects experts, SwiGLU gates within each expert — which may explain why MoE+SwiGLU (Mixtral, GPT-4 rumors) works particularly well.

## Related Concepts
[[Transformer]] [[Mixture-of-Experts|MoE]] [[GQA]] [[FlashAttention]] [[RoPE]]
