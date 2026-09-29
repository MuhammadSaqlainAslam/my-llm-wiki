---
title: "Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents"
authors: "Wonjoong Kim, Sangwu Park, Yeonjun In, Sein Kim, Dongha Lee, Chanyoung Park"
year: "2025"
arxiv: "2510.02837"
tags: [benchmarks-evaluation, agentic, tool-use, reasoning]
tldr: "TRACE — a reference-free framework for multi-dimensional evaluation of tool-augmented LLM agents that assesses full reasoning trajectories (efficiency, hallucination, adaptivity), not just final-answer accuracy. ICML 2026."
citation_count: 0
created: "2026-09-29"
---

## TL;DR
Tool-augmented agent benchmarks mostly evaluate final-answer accuracy and ignore how the agent got there. TRACE introduces a reference-free evaluation framework that assesses three trajectory dimensions: **efficiency** (avoiding unnecessary steps), **hallucination** (avoiding fabricated tool calls or results), and **adaptivity** (switching to alternative tools when a chosen one fails). An evidence bank accumulates knowledge from preceding steps, so TRACE can evaluate accurately even with small open-source LLMs. The authors also build a new meta-evaluation dataset of diverse, flawed trajectories labeled with multi-faceted performance scores, and report previously unreported observations when applying TRACE to agent trajectories. ICML 2026.

## The Problem
As tasks grow more complex and agents execute more steps, two agents can reach the same final accuracy through very different trajectories: one efficient and reliable, another hallucinating intermediate results and wasting tool calls. Answer matching cannot tell them apart. The obvious fix — compare against a ground-truth trajectory — is prohibitively expensive, because all valid trajectories would have to be annotated.

## The Idea
TRACE evaluates trajectories without annotated reference trajectories, along three dimensions:

- **Efficiency** — does the agent reach the answer without unnecessary tool calls or redundant reasoning steps?
- **Hallucination** — does it fabricate tool outputs or misreport intermediate results?
- **Adaptivity** — when a tool fails (unavailable, erroring), does it recover with an alternative rather than fail outright?

An **evidence bank** accumulates information from preceding steps, giving the evaluator LLM the full context of the trajectory so far when it scores each dimension.

## Why It Matters
- Addresses a real gap in agent evaluation: agents with similar final accuracy can differ substantially once their trajectories are examined — important for anyone deploying agents in production.
- Reference-free evaluation is practical: no exhaustive ground-truth trajectory annotation, so it scales to new benchmarks and domains.
- Fits the wiki's Benchmarks & Evaluation theme alongside [[LLM Benchmarks]] and [[Gorilla]] (API-call correctness): those score outcomes, TRACE scores the path. [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]]-style agents are the kind of tool-augmented agent it is designed to assess.

## Limitations
- Relies on an LLM as the evaluator of trajectory quality — evaluator errors can propagate into TRACE scores.
- The three dimensions may not cover every trajectory-quality axis relevant to every deployment.
- Draws on tool-augmented agent benchmarks such as GTA and m&m's — generalization to very different tool-use domains is not yet established.

## Related Concepts
[[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] · [[LLM Benchmarks]] · [[Humanity's Last Exam]] · [[Gorilla]] · [[EAPO]]
