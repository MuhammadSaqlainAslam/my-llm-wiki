---
title: "s1: Simple Test-Time Scaling"
authors: "Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto"
year: "2025"
arxiv: "2501.19393"
tags: [reasoning, test-time-scaling, inference, fine-tuning-alignment]
tldr: "Curates a 1,000-question dataset (s1K) and applies budget forcing — forcefully ending or extending a model's thinking process — so that s1-32B exceeds o1-preview on MATH and AIME24 by up to 27%, using only 26 minutes of fine-tuning on 16 H100s."
citation_count: 0
created: "2026-09-29"
---

## TL;DR
s1 achieves test-time compute scaling through two ideas: a carefully curated 1,000-question dataset (s1K) built around difficulty, diversity, and quality; and budget forcing — a technique that controls how much thinking the model does at inference time by forcefully appending an end-of-thinking token when the budget is exceeded, or appending "Wait" to extend thinking. After supervised fine-tuning Qwen2.5-32B-Instruct on s1K, the resulting s1-32B exceeds o1-preview on competition math (MATH and AIME24) by up to 27%. Scaling it further with budget forcing extrapolates beyond its no-intervention performance: 50% → 57% on AIME24. Model, data, and code are open-source.

## The Problem
OpenAI's o1 demonstrated that extra test-time compute dramatically improves reasoning, but its methodology was not publicly shared. Most replication efforts required large datasets or complex infrastructure. The question was: what is the simplest possible approach to achieve test-time scaling?

## The Idea
Two steps. First, curate **s1K** — just 1,000 questions paired with reasoning traces distilled from Gemini Flash Thinking Experimental, chosen by three criteria validated through ablations: difficulty (hard questions), diversity (many domains), and quality (clean, correct traces).

Second, fine-tune a pretrained model (Qwen2.5-32B-Instruct) on s1K in about 26 minutes on 16 H100s. At inference, apply **budget forcing**: if the model generates more thinking tokens than a budget limit, forcefully append an end-of-thinking delimiter so it transitions to answering. If the model tries to stop thinking too early, append "Wait" to make it keep going — which often leads it to double-check its answer and fix incorrect reasoning steps. This simple mechanism makes performance scale predictably with test-time compute.

## Why It Matters
- Shows that 1,000 carefully curated examples plus a simple inference trick reproduce the core test-time scaling behavior demonstrated by o1, challenging the assumption that it requires massive infrastructure or proprietary methods.
- Budget forcing is surprisingly effective: forcing the model to continue thinking lets it reconsider and correct errors rather than just padding tokens.
- Bridges the [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]] and [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] notes to the [[Test-Time-Compute]] scaling direction, one of the most active research threads of 2025. It scales compute by generating more tokens — the axis that [[Recurrent-Depth-Reasoning|recurrent-depth models]] deliberately avoid by looping in latent space instead.

## Limitations
- Budget extension is conflated with length extension — it is not always clear whether gains come from genuine reconsideration or simply more tokens.
- s1K is distilled from Gemini Flash Thinking Experimental; performance may degrade if the teacher's reasoning traces have systematic errors.
- Evaluated primarily on math competition benchmarks; generalization to other reasoning domains is less established.

## Related Concepts
[[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[VinePPO]] · [[GRPO]] · [[Kimi-k1.5|Kimi k1.5]] · [[Absolute-Zero|Absolute Zero]] · [[Test-Time-Compute]] · [[Inference-Time-Scaling]] · [[Illusion-of-Thinking|Illusion of Thinking]]
