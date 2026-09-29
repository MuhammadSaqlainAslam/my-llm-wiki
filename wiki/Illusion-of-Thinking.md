---
title: "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity"
authors: "Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar (Apple)"
year: "2025"
arxiv: ""
technical_report: "https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf"
source_type: "technical_report"
tags: [reasoning, benchmarks-evaluation, interpretability, limitations]
tldr: "Apple's systematic study showing that frontier Large Reasoning Models (LRMs) face complete accuracy collapse beyond certain problem complexities, and exhibit a counterintuitive scaling limit: reasoning effort peaks at medium difficulty then declines on hard problems."
citation_count: 0
created: "2026-09-29"
---

## TL;DR
Uses controllable puzzles (Tower of Hanoi, River Crossing, Blocks World, etc.) to systematically probe the reasoning capabilities of frontier LRMs, including Claude 3.7 Sonnet (thinking) and DeepSeek-R1, across varying complexity. Key finding: all tested LRMs face complete accuracy collapse beyond certain complexity thresholds — and counterintuitively, reasoning effort (measured in thinking tokens) rises with difficulty and then declines on the hardest problems, even with token budget remaining. Simply allowing more thinking does not grant unlimited reasoning capability. Published by Apple, June 2025 — no arXiv submission; hosted at Apple's CDN as a technical report.

## The Problem
Standard benchmarks like MMLU and MATH are saturated by frontier models, making them inadequate for probing reasoning limits. Most also conflate problem difficulty with world knowledge rather than isolating pure reasoning complexity. Understanding where and how reasoning models break down — not just where they succeed — is necessary to separate genuine from apparent capabilities.

## The Idea
Design controllable puzzles where difficulty is precisely parameterized (e.g., number of disks in Tower of Hanoi, number of agents in River Crossing), so complexity can be increased systematically while the underlying reasoning structure stays constant. Evaluate both standard LLMs and their LRM counterparts at each complexity level to map out performance curves and reasoning-effort curves simultaneously.

Three performance regimes emerge: standard LLMs are competitive or better on simple tasks, LRMs excel at medium complexity, and both ultimately collapse on hard problems.

## Why It Matters
- One of the most influential and controversial papers of 2025 — it directly challenged the "more thinking = better reasoning" narrative that accompanied the success of o1 and [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]].
- Generated substantial follow-up work both replicating and rebutting its findings; the debate clarified what test-time scaling does and does not achieve.
- Complements [[s1-Simple-Test-Time-Scaling|s1]] (test-time scaling works within a regime) and DeepSeek-R1 (which it partially critiques) — together they give a fuller picture of LLM reasoning capabilities and limits, and of [[Test-Time-Compute]] more broadly.

## Limitations
- Contested findings: replication studies reported that stepwise prompting, agentic setups, or tool augmentation partly recover performance on the hardest puzzles.
- Scope is limited to combinatorial planning puzzles — failure on Tower of Hanoi does not necessarily imply failure on scientific reasoning or open-ended tasks.
- Technical report only, without a formal peer-review venue or arXiv record.

## Related Concepts
[[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[s1-Simple-Test-Time-Scaling|s1]] · [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]] · [[Humanity's Last Exam]] · [[Kimi-k1.5|Kimi k1.5]] · [[VinePPO]] · [[Test-Time-Compute]]
