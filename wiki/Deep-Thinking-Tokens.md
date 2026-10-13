---
created: "2026-10-13"
title: "Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"
authors: "Wei-Lin Chen, Liqian Peng, Tian Tan, Chao Zhao, Blake JianHang Chen, Ziqian Lin, Alec Go, Yu Meng"
year: "2026"
arxiv: "2602.13517"
venue: "ICML 2026"
tags: [reasoning, test-time-compute, interpretability, self-consistency, logit-lens, inference]
citation_count: 0
tldr: "Token count is an unreliable proxy for reasoning effort: longer outputs often signal overthinking, not better answers. Instead, count 'deep-thinking tokens' — tokens whose internal prediction keeps being revised in the deepest layers before settling. The deep-thinking ratio correlates with accuracy (avg Pearson 0.687 vs -0.594 for token count), and Think@n uses it to drop unpromising samples early, matching or beating self-consistency at roughly half the tokens. ICML 2026."
aliases: ["Think@n", "Deep-Thinking Ratio", "DTR"]
---

# Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens

> Wei-Lin Chen, Liqian Peng, Tian Tan, Chao Zhao, Blake JianHang Chen, Ziqian Lin, Alec Go, Yu Meng, "Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens", February 2026 (arXiv:2602.13517), ICML 2026

## The Problem / Motivation

[[Test-Time-Compute]] scaling via long chain-of-thought rests on an unstated assumption: more generated tokens means more reasoning effort, and more effort means better answers. The paper starts from recent evidence that this is false — increased length does not consistently track accuracy and can signal "overthinking" that *hurts* performance. If raw length is a bad proxy, then both *measuring* reasoning effort and *allocating* test-time compute by length are on shaky ground.

## The Idea

Measure effort by **how deep in the network a token's prediction stabilizes**, not by how many tokens were written. Some tokens are decided early in the layer stack and then merely carried along; others have their predicted distribution substantially revised in the deepest layers before they converge. Those are **deep-thinking tokens**, and the fraction of them in a sequence turns out to be a much better signal of whether the reasoning is any good.

## Method

- **Layer-wise predictions.** Project each intermediate layer's hidden state through the LM head (the *logit-lens* technique) to get a distribution at every layer.
- **Settling depth.** For each generated token, compute the Jensen–Shannon divergence between each intermediate layer's distribution and the final layer's, and take the first layer where the running minimum of that divergence falls below a fixed threshold.
- **Deep-thinking token.** A token whose settling depth lands in the deep regime (layers at or beyond a fraction ρ of the depth; the paper uses ρ ≈ 0.85).
- **Deep-thinking ratio (DTR).** The proportion of a sequence's tokens that are deep-thinking tokens.
- **Think@n.** A test-time scaling strategy built on DTR: sample *n* responses, estimate each one's DTR from only a short prefix (50 tokens), keep the top half by DTR, and **majority-vote** over those. Unpromising generations can be rejected after a short prefix instead of being run to completion.

## Key Results

| Aspect | Result |
|---|---|
| Correlation with accuracy | DTR: avg Pearson r = 0.687 across eight model variants and four benchmarks |
| Baselines | Token count r = −0.594 (negative!); self-certainty 0.605; negative entropy 0.571; log-probability 0.527 |
| Think@n vs. self-consistency | Matches or exceeds standard self-consistency while significantly cutting inference cost |
| Example (AIME 2025, GPT-OSS-120B medium) | Consensus@48 92.7% at ~307.6k tokens vs. Think@48 94.7% at ~155.4k tokens (about half the cost) |
| Models | GPT-OSS-20B / 120B (low/medium/high), DeepSeek-R1-70B, Qwen3-30B-Thinking |
| Benchmarks | AIME 2024, AIME 2025, HMMT 2025, GPQA-Diamond |

## How It Fits the Wiki

This paper is the measurement counterpart to a lot of what the wiki already says about test-time compute:

- **Length is not effort.** [[s1-Simple-Test-Time-Scaling|s1]]'s budget forcing extends thinking by appending "Wait", and [[Illusion-of-Thinking|The Illusion of Thinking]] finds reasoning effort *declines* on the hardest problems. This paper gives a principled reason both can be true: token count is a weak proxy, and what matters is whether the extra tokens trigger deep computation.
- **A cheaper [[Self-Consistency]].** Think@n is self-consistency with early rejection: it keeps the voting but stops paying for samples that look shallow.
- **Depth as the unit of compute.** [[Recurrent-Depth-Reasoning|Huginn]], [[LOTUS]] and [[SMELT]] treat *layers applied* as the compute knob. Deep-thinking tokens measure how much of the *existing* depth each token actually uses — the same lens applied to a standard, non-looped model.
- **Interpretability tooling.** It uses the logit lens, the same family of early-exit probes used to study depth utilization in transformers.

## Limitations

- DTR may not be comparable across models or modes: higher reasoning-level settings showed systematically *smaller* DTR despite better accuracy, suggesting such variants shift computation from depth into sequence length.
- Requires access to intermediate hidden states and the unembedding matrix, so it does not apply to closed-source API models.
- Sensitive to hyperparameters: the settling threshold matters more than the depth fraction ρ.
- Evaluated on math and science reasoning (AIME, HMMT, GPQA-Diamond); behavior on other task types is not established here.

## Related Concepts

[[Test-Time-Compute]] · [[Inference-Time-Scaling]] · [[Self-Consistency]] · [[s1-Simple-Test-Time-Scaling|s1]] · [[Illusion-of-Thinking|Illusion of Thinking]] · [[Recurrent-Depth-Reasoning|Huginn]] · [[LOTUS]] · [[SMELT]] · [[GPQA]] · [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]]
