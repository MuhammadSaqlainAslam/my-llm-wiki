---
created: "2026-09-07"
title: "SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers"
authors: "Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, Jian Li"
year: 2026
arxiv: "2609.01343"
tags: [looped-transformer, mixture-of-experts, scaling-laws, compute-optimal, attention-sinks]
citation_count: 0
tldr: "Most looped-Transformer evaluations compare against an unlooped baseline of the same layer count, quietly giving the looped model extra FLOPs. SMELT loops the middle half of an MoE Transformer's layers twice while matching per-token FLOPs, parameters, and KV cache exactly against the baseline, and still saves 6.8-18.0% of training FLOPs on the compute-optimal frontier — with the largest gains on code tasks and longer contexts."
aliases: ["SMELT"]
---

# SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers

> Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, Jian Li, "SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers", September 2026 (arXiv:2609.01343)

## The Problem / Motivation

Looped Transformers — architectures that reuse a shared block of layers to increase effective depth — keep showing up as a promising idea (see [[Recurrent-Depth-Reasoning|Huginn]] and [[LOTUS]] elsewhere in this wiki), but the evaluations backing that promise have a quiet flaw: looping a block of layers *N* extra times, then comparing against an unlooped model with the *same number of unique layers*, is not a fair fight. The looped model does strictly more FLOPs per token. Any quality gain could just be "more compute," not "looping is architecturally better than depth." Nobody had checked whether looping still wins once FLOPs, parameter count, and KV cache are all pinned to match a non-looped baseline exactly — and nobody had checked this for Mixture-of-Experts models at all, where the FLOPs/parameter relationship is already decoupled by sparse routing.

## The Idea

Hold every resource budget fixed — per-token FLOPs, total non-embedding parameters, and KV cache size — and ask: does looping half the layers of an MoE Transformer still beat a same-budget unlooped baseline on the loss-vs-compute frontier? **SMELT** answers yes, via a specific looping recipe: take the middle half of the model's layers and loop that block twice, while shrinking/adjusting the rest of the architecture so all three budgets land exactly on the unlooped baseline's numbers. This isolates the architectural effect of looping from the confound of extra compute that plagued prior comparisons.

## Architecture / Method

```
Unlooped baseline (N layers)          SMELT (compute-matched)
─────────────────────────             ────────────────────────
 layer 1                               layer 1
 layer 2                               layer 2
   ...                                   ...
 layer N/4        ┐                    layer N/4        ┐
 layer N/4+1      │                    ┌──────────────┐ │  middle half
   ...            │  middle half       │ shared block │ │  looped twice
 layer 3N/4        ┘                   └──────┬───────┘ ┘  (iterate 2×)
   ...                                         │ (loop)
 layer N                                       ▼
                                        layer 3N/4+1
                                          ...
                                        layer N

  Budgets matched against baseline: per-token FLOPs ✓  params ✓  KV cache ✓
```

The specific choice — loop *the middle half*, and loop it *exactly twice* — is deliberate rather than arbitrary: middle layers are where prior mechanistic work on Transformer depth utilization finds the most redundancy and the most room for iterative refinement, and a fixed 2× loop factor is what makes exact budget-matching against an unlooped baseline tractable to specify and to fit a clean scaling law over.

To characterize how the two architectures scale, the authors fit a separate Chinchilla-style scaling law to each — looped-MoE and unlooped-MoE — across four model sizes up to 54B non-embedding parameters, rather than reporting a handful of point comparisons.

**Mechanistic finding.** Looking inside the trained models, the second pass through the looped middle block measurably reduces the attention-sink effect (the tendency of heads to dump disproportionate attention mass on the first few tokens rather than content) and redirects that attention toward tokens that are actually semantically relevant to the current computation. This gives a concrete mechanism for *why* re-iterating layers helps: it is not just "more compute," it is compute spent specifically un-sticking attention from a degenerate pattern.

## Key Results

| Metric | Result |
|---|---|
| Scale tested | Up to 54B non-embedding parameters, four model sizes |
| Training FLOPs saved (compute-optimal frontier) | 6.8%–18.0% vs. compute-matched unlooped baseline |
| Where gains concentrate | Largest on code tasks; grows with sequence length and number of in-context examples |
| Transfer | Advantage holds on downstream benchmarks, not just training loss |
| Mechanistic effect | Second loop iteration reduces attention-sink mass, redirects focus to content-relevant tokens |
| Comparison methodology | Chinchilla-style scaling law fit separately per architecture (looped vs. unlooped MoE), not point comparisons |

## Comparison to Prior Work

- vs. **prior looped-Transformer evaluations** (fixed-layer-count comparisons) — those conflate the benefit of looping with the extra FLOPs looping spends; SMELT is explicit about controlling FLOPs, parameters, *and* KV cache simultaneously, which is a strictly harder bar to clear.
- vs. **[[Mixture-of-Experts]] scaling in general** — MoE already decouples parameter count from per-token FLOPs via sparse routing; SMELT is (per the paper) the first work to ask whether *architectural* depth-via-looping still helps once that decoupling is already being exploited, rather than studying looping only on dense Transformers.
- vs. **[[Recurrent-Depth-Reasoning|Huginn]] (Geiping et al.)** — Huginn loops layers to give a model a variable, inference-time-controllable reasoning depth (more loops on hard problems). SMELT loops layers as a fixed architectural choice made at training time to win on the compute-optimal training frontier; there is no per-token adaptive loop count. The two papers ask different questions — "can looping create a reasoning dial?" vs. "does looping win a fair, budget-matched fight against depth?" — using the same underlying primitive.
- vs. **[[LOTUS]] (Fan, Svete & Lee)** — LOTUS loops layers and trains against explicit CoT supervision specifically to make latent reasoning competitive with explicit CoT. SMELT has no reasoning-supervision component at all; its objective is standard language-model pretraining loss, and its contribution is establishing that looping wins even under a strict, matched-budget scaling-law comparison.

## Limitations

- The specific recipe (loop the middle half, exactly twice) is one point in a larger design space (which layers to loop, how many times, whether to loop uniformly or adaptively) — the paper establishes that this point beats the unlooped baseline, not that it is optimal.
- Compute savings (6.8–18.0%) are reported at the compute-optimal frontier; realized savings on any particular hardware/serving setup would depend on how well the KV-cache-and-FLOPs-matched budget translates into matched wall-clock cost on real accelerators.
- Gains concentrate on code and long-context/many-shot settings — the paper is candid that the advantage is not uniform across all task types.

## Why It Matters

SMELT closes a methodological gap that made every prior looped-vs-non-looped Transformer comparison in the literature somewhat suspect: it shows looping still wins once you can no longer credit the win to "the looped model just did more work." That the effect survives strict FLOPs/parameter/KV-cache matching, scales cleanly via fitted Chinchilla-style laws up to 54B parameters, and comes with a mechanistic story (less attention-sink waste, more content-relevant attention) makes looping a credible general architectural lever for [[Mixture-of-Experts|MoE]] pretraining — not just a reasoning-time trick as in [[Recurrent-Depth-Reasoning|Huginn]] or [[LOTUS]]. Together, the three papers suggest 2025–2026 is the point at which "loop the layers instead of adding more of them" graduated from a curiosity into a technique worth putting real scaling-law rigor behind.

## Related Concepts

[[Mixture-of-Experts]] · [[Recurrent-Depth-Reasoning|Scaling up Test-Time Compute with Latent Reasoning (Huginn)]] · [[LOTUS]] · [[Attention sinks]] · [[Chinchilla_Scaling_Laws]] · [[Load Balancing Loss]] · [[KV Cache]]
