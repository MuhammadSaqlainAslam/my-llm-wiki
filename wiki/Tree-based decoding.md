---
created: "2026-05-20"
title: "Tree-Based Decoding"
tags: [glossary, speculative-decoding, inference, eagle, medusa, tree]
tldr: "A speculative decoding strategy where candidate continuations are organized as a tree (rather than a flat list), allowing the LLM to verify multiple branching hypotheses simultaneously via a tree attention mask. Enables exponentially more candidates per verification pass."
aliases: [tree-based decoding, tree decoding, tree speculation]
---

## TL;DR

Tree-based decoding extends speculative decoding from a linear candidate sequence to a full tree of candidates. The draft model (or heads) proposes $s$ candidates at each position, forming a tree with up to $s^K$ leaves at depth $K$. [[Tree attention]] verifies all paths in a single forward pass. Accepted tokens form the longest consistent prefix from root to any leaf.

This gives exponentially more candidates than flat speculative decoding (which proposes one linear sequence), dramatically increasing the expected accepted tokens per verification step.

Both [[Medusa]] and [[EAGLE]] use tree-based decoding: Medusa heads produce $K$ candidate sets (one per head), forming the tree; EAGLE builds the tree autoregressively using its feature draft model.

## Where It Appears in This Wiki

- [[EAGLE]] — uses tree-based decoding to maximize accepted tokens per verification pass
- [[Medusa]] — Medusa heads produce the candidate sets that form the tree
- [[Tree attention]] — the attention mechanism that verifies all tree paths in one forward pass

## Related Concepts

[[Tree attention]] · [[EAGLE]] · [[Medusa]] · [[Speculative Decoding]] · [[Draft model]]
