---
created: "2026-05-20"
title: "LLM Evaluation"
tags: [glossary, benchmarks, evaluation, metrics]
tldr: "The practice of measuring LLM capability, safety, and efficiency across standardized benchmarks. Covers knowledge (MMLU, GPQA), reasoning (MATH, GSM8K), coding (HumanEval), long-context (RULER), and agentic tasks — each measuring a different axis of model capability."
aliases: [LLM evaluation, model evaluation, benchmark evaluation]
---

## TL;DR

LLM evaluation spans multiple axes because no single benchmark captures all relevant capabilities. Key families:

- **Knowledge**: MMLU (57 subjects, 14K questions), GPQA (PhD-level science)
- **Reasoning**: MATH (competition math), GSM8K (grade school math), BBH (logical reasoning)
- **Code**: HumanEval (Python function completion), MBPP
- **Long-context**: RULER (32K–128K retrieval, reasoning), SCROLLS
- **Agentic**: SWE-bench (software engineering), WebArena, GAIA

Each benchmark tests a different capability and has different failure modes (contamination, overfitting, format sensitivity). The [[LLM Benchmarks]] note in this wiki covers the 2025 state of frontier models on these benchmarks.

## Why It Matters

- **Benchmarks are the shared language of model comparison.** Without standardized evaluation, claims about model quality are unverifiable.
- **Different benchmarks select for different capabilities.** A model best at MMLU may not be best at coding or agentic tasks.
- **Contamination is a real concern.** Models trained on data that overlaps with test sets may overfit benchmark scores without genuine capability improvements.

## Where It Appears in This Wiki

- [[LLM Benchmarks]] — the wiki's evaluation reference for frontier models

## Related Concepts

[[LLM Benchmarks]] · [[Transformer]] · [[LLaMA 2]] · [[Nemotron-3]]
