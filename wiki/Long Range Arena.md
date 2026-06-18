---
created: "2026-05-20"
title: "Long Range Arena (LRA)"
tags: [glossary, benchmark, ssm, s4, evaluation, long-range]
tldr: "A benchmark suite of 6 tasks designed to stress-test sequence models on long-range dependencies (sequences up to 16,384 tokens). S4 was the first model to solve all 6 tasks, including Path-X where every prior model scored at chance level."
aliases: [LRA, Long Range Arena, long range arena]
---

## TL;DR

Long Range Arena (Tay et al., 2021) is a benchmark with 6 tasks, each requiring capturing dependencies across thousands of tokens:

1. **ListOps** — hierarchical list parsing (up to 2K tokens)
2. **Text** — byte-level document classification (up to 4K)
3. **Retrieval** — document pair relevance (up to 4K each)
4. **Image** — sequential image classification on flattened pixels (1K)
5. **Pathfinder** — path connectivity in a noisy grid image (1K)
6. **Path-X** — same as Pathfinder but at 16K tokens — the hardest task

Path-X was a landmark: before [[S4]], **every sequence model** scored at or below 50% (chance). Transformers, LSTMs, linear attention variants — all failed. S4 scored 88.1%, the first model to solve it.

## Why It Matters

- **Path-X was a watershed.** Solving a task where all prior models fail proves that the architecture is qualitatively different, not just incrementally better.
- **LRA exposed the limits of efficient Transformers.** Many approximate attention methods scored lower than vanilla attention on LRA tasks — showing that reducing attention's complexity without preserving its recall was counterproductive.
- **It motivated the selective SSM line of research.** S4's LRA success validated the SSM approach and led directly to Mamba.

## Where It Appears in This Wiki

- [[S4]] — S4's LRA results (first to solve all 6 tasks, including Path-X) are the main empirical contribution
- [[Mamba]] — improves on S4's LRA results via selective state spaces

## Related Concepts

[[S4]] · [[State Space Model]] · [[HiPPO matrix]] · [[Mamba]] · [[Transformer]]
