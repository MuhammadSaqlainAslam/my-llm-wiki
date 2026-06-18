---
created: "2026-05-20"
title: "Lookahead Decoding"
tags: [glossary, speculative-decoding, inference, jacobi, parallel-decoding]
tldr: "An inference acceleration method that uses Jacobi iteration to simultaneously draft multiple tokens without any separate model. Runs the LLM with multiple 'lookahead' windows in parallel, accepting tokens that are self-consistent. No draft model or fine-tuning required."
aliases: [Lookahead Decoding, Jacobi decoding, lookahead decoding]
---

## TL;DR

Lookahead Decoding (Fu et al., 2024) accelerates autoregressive decoding by maintaining multiple "guess" sequences simultaneously. Starting from a draft, run the LLM with a modified attention mask to simultaneously advance multiple n-gram guesses in parallel. Accept any guesses that are self-consistent (where each token is what the LLM would have predicted given the preceding tokens). No draft model needed — just a different decoding procedure.

## Intuition

Standard decoding: generate token 1, then token 2 (conditioned on 1), then token 3 (conditioned on 1,2), etc. Lookahead breaks the sequential dependency by maintaining $W$ parallel "lookahead branches" — speculative continuations. In each step, the LLM processes all branches simultaneously, checks which branches are self-consistent (a branch is consistent if each position's prediction equals the branch's token), and accepts the first consistent $n$-gram.

The key insight: in language, many $n$-gram continuations are highly probable and self-consistent without needing to know the exact prefix. Common phrases, code patterns, and formulaic text can be verified in a single pass.

Speedup: typically 1.3–1.8× on code and formal text, lower on creative tasks. Less than [[EAGLE]] (3×) or [[Medusa]] (2.2×) but requires no fine-tuning and no draft model at all.

## Why It Matters

- **Zero infrastructure cost.** No draft model, no fine-tuning — just change the decoding procedure. Works out of the box on any causal LM.
- **Baseline for zero-overhead speculative decoding.** In [[EAGLE]]'s benchmark table, lookahead decoding is the simplest point to compare against.

## Where It Appears in This Wiki

- [[EAGLE]] — compared against as a simpler baseline in the speedup table

## Related Concepts

[[Speculative Decoding]] · [[EAGLE]] · [[Medusa]] · [[Draft model]] · [[Transformer]]
