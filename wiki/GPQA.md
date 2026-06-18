---
created: "2026-06-03"
title: "GPQA"
tags: [glossary, benchmarks, evaluation, reasoning, expert-knowledge]
tldr: "Graduate-Level Google-Proof Q&A — 448 expert-written questions in biology, chemistry, and physics that PhD holders answer at only ~65%; GPT-4 scored ~39% at launch."
---

## TL;DR
GPQA Diamond (Rein et al., 2023) is a set of expert-level multiple-choice questions in STEM that are deliberately designed to be un-Googleable. Domain experts (PhD students) answer at ~65%; non-experts with internet access score ~34%. GPT-4 scored ~39% at launch.

## Intuition
Most benchmarks can be gamed by retrieval from the training corpus. GPQA's questions require actual reasoning about narrow technical content — correct answers are not surface-level facts. "Diamond" is the hardest split. Frontier models now exceed 85-90%, so this benchmark is also starting to saturate.

## Why It Matters
- One of the few benchmarks that genuinely required expert reasoning at launch
- Diamond variant is preferred for frontier model comparisons
- Supplements [[MMLU]] with genuine depth vs. breadth; [[Humanity's Last Exam]] extends further

## Related Concepts
[[MMLU]] · [[Humanity's Last Exam]] · [[LLM Benchmarks]] · [[LLM evaluation]]
