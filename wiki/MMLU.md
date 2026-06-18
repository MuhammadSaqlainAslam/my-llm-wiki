---
created: "2026-06-03"
title: "MMLU"
tags: [glossary, benchmarks, evaluation, knowledge]
tldr: "Massive Multitask Language Understanding — 57-subject multiple-choice test covering STEM, humanities, and social sciences; a standard knowledge breadth benchmark since 2021."
---

## TL;DR
MMLU (Hendrycks et al., 2021) tests a model's broad factual knowledge across 57 academic subjects, from high school biology to professional law and medicine. Each question is 4-choice multiple-choice; random baseline is 25%.

## Intuition
MMLU is a proxy for the breadth of knowledge absorbed during pretraining. Models score well by having memorized a wide range of factual content. GPT-4 broke 85% at launch; frontier models now routinely exceed 90%. Increasingly seen as saturated — harder variants (MMLU-Pro, GPQA) are now preferred.

## Why It Matters
- Cheapest and most widely-cited knowledge breadth benchmark — nearly every LLM paper reports it
- Saturating: GPT-4-class models score >85%; useful mainly for distinguishing smaller/older models
- MMLU-Pro adds harder reasoning variants; [[Humanity's Last Exam]] targets PhD-level difficulty

## Related Concepts
[[GPQA]] · [[Humanity's Last Exam]] · [[LLM Benchmarks]] · [[LLM evaluation]]
