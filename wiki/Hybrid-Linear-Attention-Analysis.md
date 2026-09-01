---
created: "2026-09-01"
title: "A Systematic Analysis of Hybrid Linear Attention"
authors: "Dustin Wang, Rui-Jie Zhu, Steven Abreu, Yong Shan, Taylor Kergan, Yuqi Pan, Yuhong Chou, Zheng Li, Jibin Wu, Ge Zhang, Wenhao Huang, Jason Eshraghian"
year: 2025
arxiv: "2507.06457"
tags: [linear-attention, hybrid-architecture, empirical-study, gated-deltanet, hgrn2, recall]
citation_count: 0
tldr: "Trains and open-sources 72 models (340M and 1.3B params, six linear-attention variants × five hybridization ratios) to answer a question nobody had actually tested: does a stronger standalone linear-attention model make a stronger hybrid? Answer: no. Selective gating, hierarchical recurrence, and controlled forgetting predict hybrid quality better than standalone-model strength does; HGRN-2 or Gated DeltaNet at a 3:1–6:1 linear-to-full ratio is the recommended default."
aliases: ["Hybrid Linear Attention Analysis", "Systematic Analysis of Hybrid Linear Attention"]
---

# A Systematic Analysis of Hybrid Linear Attention

> Dustin Wang, Rui-Jie Zhu, Steven Abreu, Yong Shan, Taylor Kergan, Yuqi Pan, Yuhong Chou, Zheng Li, Jibin Wu, Ge Zhang, Wenhao Huang, Jason Eshraghian, "A Systematic Analysis of Hybrid Linear Attention", July 2025, revised June 2026 (arXiv:2507.06457)

## TL;DR

Hybrid models that interleave full attention with a linear-attention mechanism ([[Gated_DeltaNet_(Yang_et_al._2025)|Gated DeltaNet]], [[HGRN2]], [[Griffin]]'s RG-LRU, and others) are everywhere now — but almost every paper that builds one picks its linear-attention component somewhat arbitrarily, then spends its ablation budget on the *ratio* of linear-to-full layers instead of on *which* linear mechanism to use. This paper asks the obvious question nobody had answered empirically: does a linear-attention variant that's strong on its own (standalone, no attention at all) also make a strong *hybrid*? The authors train 72 models — 36 at 340M params / 20B tokens, 36 at 1.3B params / 100B tokens, spanning six linear-attention variants across five hybridization ratios — and find the answer is **no**. Standalone strength doesn't transfer. What predicts hybrid quality instead is **selective gating, hierarchical recurrence, and controlled forgetting** — properties like those in HGRN-2 and Gated DeltaNet, which the paper recommends pairing with a 3:1 to 6:1 linear-to-full ratio to reach Transformer-level recall.

## The Problem / Motivation

By 2025 there were many published hybrid Transformer/linear-attention architectures ([[Jamba]], [[Zamba]], [[Samba]], [[Griffin]], and more), each picking a specific linear-attention layer — a vector-recurrence model, a gated variant, [[Gated_DeltaNet_(Yang_et_al._2025)|Gated DeltaNet]], [[HGRN2]] — and a specific mixing ratio. But cross-paper comparison is nearly impossible: different training data, different scales, different full-attention placement schemes. The implicit assumption baked into most of this work is that the linear-attention component barely matters as long as you get the ratio right, or conversely, that whichever linear model is strongest standalone will also make the best hybrid. Neither assumption had actually been tested in a controlled setting.

## The Idea

Hold everything else fixed — data, scale, training recipe, full-attention placement — and vary only two things: **which linear-attention mechanism** and **what fraction of layers are linear vs. full attention**. Train enough models across that 2D grid to see whether standalone strength predicts hybrid strength, and if not, what does.

## Architecture / Method

The experimental grid:

```
                     Hybridization ratio (linear : full attention layers)
                 1:1      2:1      3:1      4:1      6:1
Linear variant ┌──────┬──────┬──────┬──────┬──────┐
  Vector recur.│  •   │  •   │  •   │  •   │  •   │
  Gated (v1)   │  •   │  •   │  •   │  •   │  •   │
  Gated DeltaNet│ •   │  •   │  •   │  •   │  •   │
  HGRN-2       │  •   │  •   │  •   │  •   │  •   │
  Griffin RG-LRU│ •   │  •   │  •   │  •   │  •   │
  (6th variant)│  •   │  •   │  •   │  •   │  •   │
                └──────┴──────┴──────┴──────┴──────┘
   × 2 scales (340M/20B tokens, 1.3B/100B tokens) = 72 models total
```

Each of the 72 models is evaluated on standard language-modeling perplexity and on synthetic/real recall benchmarks (the tasks that expose whether a fixed-size recurrent state has lost information a full-attention layer would have kept). The standalone (pure linear, no attention) version of each of the six variants is also trained and evaluated, so the paper can directly correlate "standalone strength" against "hybrid strength" for the same underlying mechanism.

## Key Results

| Finding | Detail |
|---|---|
| Standalone strength → hybrid strength? | **No correlation.** A linear-attention variant that's strong standalone is not reliably strong once hybridized with attention. |
| What *does* predict hybrid quality | Selective gating, hierarchical recurrence, and controlled forgetting — architectural properties, not standalone benchmark scores. |
| Best-performing variants in hybrids | **HGRN-2** and **Gated DeltaNet** |
| Recommended hybridization ratio | **3:1 to 6:1** (linear : full attention layers) to reach Transformer-level recall efficiently |
| Scale of the study | 72 models released open-source: 36 @ 340M params (20B tokens) + 36 @ 1.3B params (100B tokens), 6 linear variants × 5 ratios |

## Comparison to Prior Work

- vs. **individual hybrid papers ([[Jamba]], [[Zamba]], [[Samba]], [[Griffin]])** — each of those papers validates one linear-attention choice at one scale with one training recipe; this paper is the first controlled cross-comparison holding everything else fixed, which is precisely what makes its negative result (standalone ≠ hybrid strength) credible rather than an artifact of confounded comparisons.
- vs. **[[Priming]]** (also in this wiki, arXiv:2605.08301) — Priming makes SSM-type comparison at frontier scale *cheap* by transferring from a pretrained Transformer instead of training from scratch; this paper takes the opposite approach — train everything from scratch, but at a scale (340M–1.3B) where 72 full runs is still affordable — and focuses specifically on linear-attention variants rather than including full SSMs like Mamba-2 in the head-to-head.
- vs. **the general SSM/linear-attention literature (S4, RWKV, RetNet, Mamba)** in Theme II of this wiki — most of those papers report results for their *own* mechanism in isolation; this paper's contribution is the comparison itself, not a new mechanism.

## Limitations

- Largest scale tested is 1.3B parameters / 100B tokens — an order of magnitude or more below frontier hybrid models (tens of billions of parameters, trillions of tokens). The ranking could shift at larger scale, though the paper's own framing suggests the *properties* that matter (gating, hierarchy, forgetting) are more likely to be scale-invariant than any specific variant's raw score.
- Six linear-attention variants and five ratios is a large grid, but it's not exhaustive — mixture ratios that vary *by layer depth* (rather than a single global ratio) aren't part of the sweep.
- Recall benchmarks used to evaluate the fixed-size-state weakness are necessarily synthetic proxies; real-world long-context recall failures can look different.

## Why It Matters

This paper replaces folk wisdom with data for one of the most consequential design decisions in modern efficient-architecture work: which linear-attention mechanism to hybridize with attention, and in what ratio. Its central finding — that a linear model's standalone strength doesn't predict its usefulness inside a hybrid — is a genuine surprise that should change how future hybrid papers justify their component choices, and its recommendation (HGRN-2 or Gated DeltaNet, 3:1–6:1 ratio) gives practitioners a concrete, empirically-grounded default instead of copying whatever the last paper did. The 72 open-sourced checkpoints also make this a reusable benchmark suite for anyone proposing a new linear-attention mechanism and wanting to know whether it will actually help in the hybrid setting that matters in production.

## Related Concepts

[[Gated_DeltaNet_(Yang_et_al._2025)|Gated DeltaNet]] · [[HGRN2]] · [[Griffin]] · [[Mamba]] · [[Jamba]] · [[Zamba]] · [[Samba]] · [[Priming]]
