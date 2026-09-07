---
created: "2026-09-07"
title: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"
authors: "Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein"
year: 2025
arxiv: "2502.05171"
venue: "NeurIPS 2025"
tags: [looped-transformer, recurrent-depth, latent-reasoning, test-time-compute, reasoning]
citation_count: 0
tldr: "Scales test-time compute by looping a recurrent block of layers at inference time instead of generating more chain-of-thought tokens. A 3.5B-parameter model (nicknamed Huginn) trained on 800B tokens improves reasoning benchmark performance up to the compute equivalent of a 50B-parameter model, purely by iterating longer in latent space, with no specialized CoT training data required."
aliases: ["Huginn", "Recurrent Depth", "Latent Reasoning Recurrent Depth"]
---

# Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach

> Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein (Maryland / LLNL), "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", February 2025 (arXiv:2502.05171), NeurIPS 2025

## The Problem / Motivation

The dominant recipe for spending more inference-time compute on a hard problem is chain-of-thought: make the model emit more tokens, so more forward passes (and more attention over a growing context) happen before the final answer. This works, but it ties "thinking harder" to "writing more words" — it needs specialized CoT training data, burns context-window budget, and can only represent reasoning that is easily expressible in natural language. Anything a model might do internally that doesn't verbalize cleanly (holding several partial hypotheses, iteratively refining a numeric estimate, tracking state that isn't naturally sentence-shaped) has no outlet in this paradigm.

## The Idea

Give the model a second, orthogonal knob for test-time compute: instead of generating more tokens, iterate the same block of layers more times per token before moving on. A **recurrent** ("looped") Transformer block is unrolled to an arbitrary depth at inference time — depth becomes a dial the model can turn up on hard problems and down on easy ones, entirely in latent space, without ever touching the visible output.

This is a direct architectural embodiment of implicit / latent reasoning: whatever computation chain-of-thought spends on committing intermediate thoughts to tokens, recurrent depth instead spends on refining a hidden state that never needs to be verbalized or trained against explicit reasoning traces.

## Architecture / Method

The model splits into three pieces:

- **Prelude (P)** — a shallow stack of ordinary Transformer layers that embeds the input tokens into a latent state.
- **Core (R)** — a single recurrent block of layers, applied repeatedly. Each iteration takes the current hidden state (plus the original embedded input, injected at every step) and updates it — this is the actual "thinking" loop, unrolled for as many steps as the compute budget allows.
- **Coda (C)** — a shallow stack of layers that reads the final hidden state after looping stops and produces the output.

```
tokens → [ Prelude P ] → latent state h₀
                              │
                    ┌─────────▼─────────┐
                    │   Core block R    │  ◄── loop r times
                    │  h_i = R(h_{i-1}, embedded input) │
                    └─────────┬─────────┘
                              │  (loop count r chosen at inference time)
                              ▼
                       latent state h_r
                              │
                        [ Coda C ] → output tokens
```

Because the same weights are reused every iteration, adding depth at test time costs no extra parameters — only extra compute. The number of loops r is a free inference-time choice, not fixed at training time, which is what lets a single trained model trade latency for accuracy on demand.

Training a model to converge to a useful fixed point after variably many iterations (rather than collapsing to a degenerate loop or diverging) is itself the hard part; the paper trains this proof-of-concept — nicknamed **Huginn** — at 3.5B parameters on 800B tokens.

## Key Results

| Aspect | Result |
|---|---|
| Scale | 3.5B parameters, 800B training tokens |
| Effect of more test-time looping | Reasoning-benchmark performance improves, sometimes dramatically, as loop count increases |
| Compute ceiling tested | Improvements continue up to compute equivalent to a ~50B-parameter model |
| Training data requirement | None specialized — no chain-of-thought traces needed, unlike explicit-CoT reasoning models |
| Context window | Works with small context windows, since reasoning happens in recurrent depth, not in generated tokens |
| Free side effects at inference | Per-token adaptive compute (loop more on hard tokens, less on easy ones), (self-)speculative decoding, and KV-cache sharing across loop iterations — all fall out of the recurrent structure without extra engineering |

## Comparison to Prior Work

- vs. **explicit chain-of-thought** (the dominant test-time-scaling recipe, e.g. o1/R1-style RLVR reasoning) — CoT scales compute by writing more tokens into the context; recurrent depth scales compute by iterating hidden state, with no requirement that the reasoning be verbalizable or that training data contain explicit reasoning traces.
- vs. classic weight-tied / **Universal Transformer**-style looping — the architectural mechanism (share weights, loop the block) is not new in isolation, but this paper is the first to push it to billions of parameters and hundreds of billions of tokens and show the resulting latent reasoning transfers to real benchmark gains, plus documents the inference-time side benefits (adaptive compute, self-speculative decoding, KV sharing).
- vs. **[[SMELT]]** — SMELT also loops Transformer layers, but for a different purpose: it loops during *training* under a fixed compute budget to improve the loss-vs-compute frontier of a static-depth model, not to give the model a variable, inference-time-controllable reasoning dial. SMELT's looped model always runs the same number of loop iterations at inference; Huginn's core contribution is making that iteration count a runtime choice.
- vs. **[[LOTUS]]** (Fan, Svete & Lee) — LOTUS also loops Transformers for latent reasoning, but adds explicit parallel supervision on latent positions using gold CoT tokens to close the latent-vs-explicit CoT performance gap; Huginn trains without any such supervision, at the cost of a less directly interpretable latent trajectory.

## Limitations

- Demonstrated at 3.5B/800B token proof-of-concept scale — not yet a frontier-scale production model, so it is unclear how the loop-count-vs-accuracy tradeoff holds at 100B+ parameters.
- Training a model to have a well-behaved, non-degenerate fixed point under a variable number of recurrent iterations is nontrivial and not yet a solved, standardized recipe the way explicit-CoT RLVR training has become.
- Latent reasoning traded away is also latent *interpretability* — unlike explicit CoT, there is no visible trace of what the extra loops actually computed, which matters for auditing and debugging model reasoning.

## Why It Matters

This paper opens up a second axis for test-time compute scaling that is orthogonal to everything chain-of-thought and [[RLVR]]-trained reasoning models do: depth-of-computation-per-token instead of length-of-output. It reframes "thinking" as something that can happen entirely inside the residual stream, unlocking reasoning that doesn't have to be representable in words, and it gets several systems benefits (adaptive per-token compute, self-speculative decoding, KV-cache sharing) essentially for free as a side effect of the recurrent architecture. It is one of the founding papers — alongside [[LOTUS]] and [[SMELT]] — of a small but growing 2025–2026 cluster of work asking whether looping a Transformer's layers, rather than deepening or widening it, is a better lever for both reasoning quality and training/inference efficiency.

## Related Concepts

[[Test-Time-Compute]] · [[Inference-Time-Scaling]] · [[LOTUS]] · [[SMELT]] · [[RLVR]] · [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]] · [[Speculative Decoding]] · [[KV Cache]]
