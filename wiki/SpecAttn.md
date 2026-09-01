---
created: "2026-06-11"
title: "Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention"
authors: "Yikang Yue, Yuqi Xue, Jian Huang"
year: 2026
arxiv: "2602.07223"
tags: [speculative-decoding, sparse-attention, kv-cache, inference]
citation_count: 2
tldr: "Self-speculative decoding where the verification pass's full attention scores double as a free oracle for which KV entries matter next — reusing them to build the sparse-attention mask for the following draft phase, instead of running a separate KV-selection algorithm. 1.25x-2.81x faster than default vLLM."
aliases: ["Vegas", "SpecAttn"]
---

# Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention

> Yikang Yue, Yuqi Xue, Jian Huang (University of Illinois Urbana-Champaign), "Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention", February 2026, ICML 2026 (arXiv:2602.07223)

*Note: this paper was originally submitted under the title "SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding" and renamed to "Vegas" on its May 2026 revision — this note (and its filename) keeps the original `SpecAttn` alias so existing links resolve.*

## TL;DR

Self-speculative decoding with sparse attention already speeds up long-context inference losslessly — draft cheaply with sparse attention over a subset of the [[KV Cache]], then verify with full attention. But prior methods select *which* KV entries matter for the sparse draft pass using a standalone selection algorithm, run as extra overhead. Vegas notices something simpler: the verification pass already computes full attention, which means it already computes the exact attention score — the true "criticality" — of every KV entry, as a free byproduct. Reuse those scores directly to build the sparse-attention mask for the *next* drafting phase, and the standalone KV-selection step disappears entirely. Result: 1.25×–2.81× faster decoding than default vLLM, and 1.15×–1.29× faster than prior state-of-the-art sparse-attention self-speculative methods.

## The Problem / Motivation

Long-context LLM inference is bottlenecked by the growing [[KV Cache]] — this wiki's [[KV Cache Optimization]] survey covers the general landscape of fixes. One specific fix, self-speculative decoding with sparse attention, works like ordinary [[Speculative Decoding]] except drafting and verification use the *same* model: the draft pass runs cheap sparse attention over only a subset of cached keys/values, and the verification pass runs full attention to check the drafts, exactly like a target model checking a smaller draft model's proposals.

The catch: someone has to decide which KV entries the sparse draft pass should attend to. Prior approaches bolt on a standalone KV-selection algorithm to make that decision — extra compute, extra complexity, and crucially, a selection that's only a *guess* at which entries will matter, made without the benefit of actually running full attention over them.

## The Idea

The verification pass, because it runs full attention, has already computed the ground-truth attention weight of every KV entry for the tokens it's checking. That's precisely the "criticality" signal a KV-selection algorithm is trying to estimate — except verification computes the *exact* value as a side effect of work it was doing anyway, not an estimate. Vegas's insight: treat this as a free oracle. Take the attention scores computed during verification, and use them directly to build the sparse-attention mask for the *next* round of drafting — no separate selection algorithm needed.

## Architecture / Method

```
Round t:
  ┌────────────────────────────┐
  │ Draft phase (sparse attn)   │  attends only to KV entries selected
  │ propose tokens t_1..t_N      │  by the ORACLE from round t-1's verify
  └────────────────────────────┘
              │
              ▼
  ┌────────────────────────────┐
  │ Verify phase (full attn)    │  checks t_1..t_N against the target
  │ — computes EXACT attention  │    model's true distribution
  │   score for every KV entry  │
  │   as a side effect          │
  └────────────────────────────┘
              │
              │ "free oracle": reuse these exact scores
              ▼
  ┌────────────────────────────┐
  │ Build sparse-attention mask │  for round t+1's draft phase —
  │ for NEXT draft phase        │  no standalone KV-selection algorithm
  └────────────────────────────┘
              │
              ▼
Round t+1: draft phase uses the oracle-selected KV entries ...
```

This closes a loop that prior sparse-attention self-speculative methods left open: verification and KV-selection used to be two separate computations estimating overlapping information; Vegas makes verification *produce* the selection information it needs for the next round, at essentially no added cost.

## Key Results

| Comparison | Speedup |
|---|---|
| Vegas vs. default vLLM | **1.25× – 2.81×** decoding throughput |
| Vegas vs. prior state-of-the-art sparse-attention self-speculative decoding | **1.15× – 1.29×** decoding throughput |

The paper also reports improved draft-token acceptance rate (because the sparse-attention mask is built from ground-truth criticality rather than an estimate) alongside low KV-selection overhead (because there's no standalone selection step left to run).

## Comparison to Prior Work

- vs. **standalone-KV-selection sparse-attention self-speculative decoding** — those methods spend extra compute estimating which KV entries matter; Vegas gets that information for free from verification's own full-attention pass, improving both acceptance rate and overhead simultaneously.
- vs. **classic two-model [[Speculative Decoding]] (Leviathan 2023, Chen 2023)** — classic speculative decoding uses a separate, smaller draft model; Vegas is self-speculative — one model plays both roles, with sparse vs. full attention (not model size) as the speed/accuracy lever, similar in spirit to how [[VeriCache]] (also newly added to this wiki) uses compressed vs. full KV cache as its lever rather than a separate draft model.
- vs. **[[VeriCache]]** — both papers reuse information produced during a "verification-equivalent" step to make a cheaper subsequent step better/faster (Vegas reuses attention scores for KV selection; VeriCache reuses full-cache verification to guarantee losslessness). They target different points on the speed/accuracy curve: Vegas keeps drafting lossy-by-design (sparse attention, verified and corrected like any speculative method) at the token level, while VeriCache targets exact-output guarantees.

## Limitations

- The "free oracle" only exists because verification runs full attention — if a system chooses partial/approximate verification for extra speed, the oracle signal degrades and the technique's core mechanism weakens.
- Gains are demonstrated in a long-context, single-model self-speculative setting; the technique doesn't directly extend to two-model speculative decoding, where the verifier's attention pattern isn't necessarily informative about the (different) draft model's needs.
- Evaluated on top of vLLM specifically; portability of the exact speedup numbers to other serving stacks with different attention-kernel implementations isn't established.

## Why It Matters

Vegas is a small, sharp idea with an outsized practical payoff: a computation the system was already doing (verification's full attention) turns out to fully subsume a computation people were running separately (KV-criticality estimation for sparse drafting). That's the same kind of "stop computing something you already have" insight that makes [[FlashAttention]] and [[KV Cache]] reuse valuable elsewhere in this wiki — it's not a new capability, it's removing redundant work, which is often where the largest and most durable systems wins come from.

## Related Concepts

[[Speculative Decoding]] · [[GQA]] · [[FlashAttention]] · [[Multi-Token Prediction]] · [[KV Cache]] · [[Sliding Window Attention]] · [[VeriCache]]
