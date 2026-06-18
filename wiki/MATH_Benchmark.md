---
created: "2026-06-03"
title: "MATH Benchmark"
tags: [glossary, benchmarks, evaluation, math, reasoning]
tldr: "Competition mathematics benchmark — 12,500 problems from AMC, AIME, and Olympiad levels; GPT-4 scored ~52% at launch, frontier models now exceed 85%."
---

## TL;DR
MATH (Hendrycks et al., 2021) contains 12,500 competition math problems from AMC 10/12, AIME, and national olympiads at 5 difficulty levels. Solutions require multi-step algebraic, geometric, or combinatorial reasoning. GPT-4 scored ~52% at launch; frontier models now top 85%.

## Intuition
Unlike MMLU (factual recall), MATH requires step-by-step derivation. The hardest Level 5 problems rival early AIME problems. Because problems have short, verifiable numeric/algebraic answers, evaluation is exact and contamination-resistant. Increasingly supplemented by harder benchmarks like [[Humanity's Last Exam]] and IMOAnswerBench.

## Why It Matters
- First widely-used symbolic-reasoning benchmark for LLMs
- Concrete, exact answers make scoring reliable
- Performance jumped from ~10% (GPT-3) to ~85%+ (frontier 2025) — useful for tracking generational progress

## Related Concepts
[[GPQA]] · [[MMLU]] · [[Humanity's Last Exam]] · [[LLM Benchmarks]] · [[LLM evaluation]]
