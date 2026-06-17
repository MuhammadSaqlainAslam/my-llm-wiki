---
title: "ARC-AGI Benchmark"
aliases: ["ARC", "Abstraction and Reasoning Corpus", "ARC-AGI"]
year: 2019
tags: [benchmark, reasoning, agi, evaluation, few-shot, stub]
tldr: "François Chollet's 2019 benchmark for measuring abstract reasoning rather than pattern memorization: each task shows 3-5 input/output grid examples and asks the model to infer the hidden rule and apply it to a new grid — humans solve ~84%, best LLMs reach ~50-60%."
---

## TL;DR
ARC presents colored pixel grids. Each task has a handful of demonstrations showing a transformation rule — "reflect about the vertical axis", "fill enclosed regions", "extend the pattern" — and asks the solver to apply it to a test grid. The key property is that each task is unique: you cannot memorize the answer from training data. You must perform genuine inductive generalization.

## Intuition
Imagine IQ-test-style matrix puzzles, but grid-shaped. A human who has never seen the task before will usually solve it in seconds — the rule is always "simple" by some reasonable human prior. LLMs that have ingested the entire internet often fail because the rule isn't statistically predictable from prior text; it requires truly composing primitives from scratch.

## Why It Matters
- Designed to be resistant to dataset memorization — new test tasks are always held out
- Highlights the gap between statistical pattern matching and core knowledge reasoning
- Used as a benchmark in looped reasoning models: [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers|FPRM]], HRM, TRM claim progress here
- ARC-AGI-2 (2025) raised the bar further; top models still far below human level

## Current State (2025-2026)
Best results come from hybrid approaches: search-augmented programs + LLM scoring. Pure LLMs (even GPT-4o, Claude 3.7) plateau in the 50–60% range on ARC-AGI-1; ARC-AGI-2 is harder still.

## See Also
[[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers]] · [[Test-Time Compute Scaling]] · [[Chain-of-Thought Prompting]] · [[Adaptive Computation Time]]
