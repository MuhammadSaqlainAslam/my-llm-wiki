---
created: "2026-09-07"
title: "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers"
authors: "Ying Fan, Anej Svete, Kangwook Lee"
year: 2026
arxiv: "2606.31779"
tags: [looped-transformer, latent-reasoning, chain-of-thought, recurrent-depth, reasoning]
citation_count: 0
tldr: "Latent chain-of-thought reasoning (thinking in hidden states instead of tokens) has historically fallen further behind explicit CoT as models scale past 1B parameters. LOTUS closes that gap by looping Transformer blocks and supervising K parallel latent positions across R loop iterations with gold CoT-step targets — the first latent-CoT method to match explicit CoT at 3B scale, with 2.5–6.9× lower thought-phase latency."
aliases: ["LOTUS", "Looped Transformers with parallel supervision on latents"]
---

# Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers

> Ying Fan, Anej Svete, Kangwook Lee, "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers", June 2026 (arXiv:2606.31779)

## The Problem / Motivation

Explicit chain-of-thought writes out reasoning steps as tokens: slow (every step costs a forward pass and grows the context) but effective, and it keeps improving as models scale. **Latent** CoT tries to get the speed benefit by reasoning inside hidden states instead — no intermediate tokens, so no extra context growth and much lower thought-phase latency. The catch, which this paper states plainly: existing latent-CoT methods underperform explicit CoT once models pass roughly 1B parameters, and the gap *widens* with scale rather than closing. Latent reasoning has been a promising idea that gets worse, relatively, exactly when it would matter most.

## The Idea

**LOTUS** (**Lo**oped **T**ransformers with parallel s**u**pervision on late**n**t**s** — reading the acronym backward from the paper's own expansion) attacks this with two ingredients used together:

1. **Loop the Transformer.** Reuse a block of layers across R iterations to increase effective computational depth without adding parameters — the same architectural move as [[Recurrent-Depth-Reasoning|Huginn]] and [[SMELT]], applied here specifically to latent reasoning.
2. **Supervise the latents directly and in parallel.** Rather than letting the looped hidden states drift unconstrained (the failure mode that causes prior latent-CoT methods to degrade with scale), LOTUS processes K latent positions in parallel across the R loop iterations and applies cross-entropy loss on those latent positions against the token targets of a gold explicit CoT trace. The model is trained to make its latent computation *track* what an explicit CoT would have written, without ever having to emit it.

The key claim the paper's ablations support is that neither ingredient alone is enough — looping without latent supervision drifts into unconstrained, harder-to-scale representations (the pre-existing latent-CoT failure mode), and parallel latent supervision without a looped/recurrent-depth architecture doesn't have enough effective computation to match explicit CoT's step-by-step depth.

## Architecture / Method

```
Explicit CoT                          LOTUS (latent CoT)
─────────────                         ───────────────────
 prompt                                 prompt
   │                                      │
   ▼                                      ▼
[Transformer] → token → [Transformer]   [Looped Transformer block, R iterations]
   → token → ... → answer                 K latent positions per iteration
   (one forward pass per                  ── CE loss vs. gold CoT-step tokens ──
    reasoning token, tokens                    (supervision only, not emitted)
    visible + in context)                      │
                                                ▼
                                          [Coda] → answer
                                          (2.5–6.9× lower thought-phase latency)
```

Because the K latent positions are supervised in parallel against the gold CoT's tokens (rather than the model being left to discover a useful latent trajectory on its own), the resulting latent space stays interpretable — the paper reports that it recovers the gold reasoning steps and even surfaces alternative valid solution paths, which is not typically true of prior latent-CoT approaches.

## Key Results

| Comparison | Result |
|---|---|
| LOTUS vs. explicit CoT at 3B scale | First latent-CoT method to match explicit CoT performance at this scale |
| Thought-phase latency | 2.5×–6.9× lower than explicit CoT |
| Latent space interpretability | Recovers gold reasoning steps; surfaces alternative valid solutions |
| Ablation: looping only (no parallel latent supervision) | Underperforms — architecture alone insufficient |
| Ablation: parallel latent supervision only (no looping) | Underperforms — supervision alone insufficient |
| Conclusion | Both looped architecture and parallel latent supervision are necessary |

## Comparison to Prior Work

- vs. **prior latent-CoT methods** (e.g., latent-thought / continuous-CoT approaches that reason in hidden states without token-level supervision) — those methods' quality gap to explicit CoT widens with scale; LOTUS's parallel latent supervision against gold CoT tokens is specifically the fix that keeps latent reasoning competitive as models grow.
- vs. **[[Recurrent-Depth-Reasoning|Huginn]] (Geiping et al.)** — Huginn also loops a Transformer core to scale test-time compute in latent space, but trains without explicit supervision on the latent trajectory, trading interpretability for not needing gold CoT traces at all. LOTUS instead uses gold CoT-step tokens as a training signal on the latents, buying scale-robustness and interpretability at the cost of requiring CoT-annotated training data.
- vs. **[[SMELT]]** — SMELT loops layers purely to improve the compute-optimal training frontier of a Mixture-of-Experts language model (matched FLOPs/params/KV-cache vs. an unlooped baseline); LOTUS loops layers specifically as a reasoning mechanism trained against explicit CoT targets. Different motivation, same underlying "reuse a block of layers across iterations" architectural primitive.
- vs. **explicit chain-of-thought** — LOTUS keeps explicit CoT's supervision signal (gold reasoning-step tokens) but moves the actual computation into latent iterations instead of the visible token stream, cutting thought-phase latency 2.5–6.9× while matching accuracy at 3B scale.

## Limitations

- Requires gold CoT-step token supervision during training — unlike fully latent methods (or Huginn), it cannot bootstrap latent reasoning ability from data that lacks explicit reasoning annotations.
- Demonstrated matching explicit CoT "at 3B scale" — whether the closed gap holds, widens back, or shrinks further at frontier (100B+) scale is not yet established.
- The K-parallel-latent-positions-across-R-iterations design adds architectural and training complexity relative to either pure looping or pure explicit CoT.

## Why It Matters

LOTUS is direct evidence that the recurring failure mode of latent reasoning — falling further behind explicit CoT as scale increases — is not an inherent property of "thinking in hidden states" but an artifact of under-constrained training. Supervising the loop's intermediate latents against real reasoning traces, rather than leaving them to drift, is enough to close the gap at the scale tested. Combined with [[Recurrent-Depth-Reasoning|Huginn]]'s unsupervised recurrent-depth approach and [[SMELT]]'s efficiency-motivated looping, LOTUS is part of a 2025–2026 cluster establishing that looping Transformer layers — long treated as a curiosity next to scaling depth or width — is becoming a serious, general-purpose lever for both reasoning quality and inference latency.

## Related Concepts

[[Recurrent-Depth-Reasoning|Scaling up Test-Time Compute with Latent Reasoning (Huginn)]] · [[SMELT]] · [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]] · [[Test-Time-Compute]] · [[Inference-Time-Scaling]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]]
