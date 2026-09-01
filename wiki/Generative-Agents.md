---
created: "2026-08-14"
title: "Generative Agents: Interactive Simulacra of Human Behavior"
authors: "Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein"
year: 2023
arxiv: "2304.03442"
tags: [agents, memory, simulation, planning, reflection, multi-agent]
citation_count: 5297
tldr: "25 LLM-powered agents live in a simulated town (Smallville), each with a memory stream, a retrieval mechanism, a reflection mechanism, and a planning mechanism. From these four ingredients alone, believable emergent social behavior appears — one agent decides to throw a Valentine's Day party, and the invitation spreads organically through the town's social graph over simulated days."
aliases: ["Generative Agents", "Smallville"]
---

# Generative Agents: Interactive Simulacra of Human Behavior

> Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein, "Generative Agents: Interactive Simulacra of Human Behavior", UIST 2023 (arXiv:2304.03442)

## TL;DR

Video game NPCs and social simulations have always needed either hand-scripted behavior trees (rigid, doesn't generalize) or narrow trained models (brittle, doesn't compose). Neither produces characters that remember what happened yesterday, form opinions about it, and act on those opinions coherently over the following days.

This paper's answer: wrap an LLM in an architecture with **memory**, not just a prompt. A generative agent keeps an append-only **memory stream** of everything it perceives, **retrieves** the most relevant/recent/important memories when deciding what to do next, periodically **reflects** on those memories to form higher-level opinions about itself and others, and uses all of this to **plan** its day and **react** when things change. Twenty-five of these agents were dropped into a simulated town called Smallville. Nobody scripted a Valentine's Day party — one agent (Isabella Rodriguez) decided to host one, mentioned it to people she ran into, and by the end of the simulation multiple agents had heard about it, made plans to attend, and some had even asked each other as dates. That's the headline result: coherent, multi-day, socially-emergent behavior from four architectural ingredients bolted onto an off-the-shelf LLM.

## The Problem / Motivation

Believable agents — characters whose behavior looks plausibly human to an observer, not merely competent at a benchmark — are hard to get from either classical approach. Hand-authored finite-state-machines and behavior trees in games are believable only within the narrow set of situations their author anticipated; they don't generalize to novel combinations of events. Trained RL/ML agents optimize a narrow objective and don't produce open-ended, contextually appropriate social behavior (deciding to invite a coworker to a party because you like them and know they're free that evening).

LLMs are good at producing plausible-sounding responses to a single prompt, but a single prompt has no persistent state — ask an LLM-backed character what it did yesterday and, without an explicit memory mechanism, it has no way to know. The question this paper asks: can an architecture built around an LLM, but not just an LLM, produce agents that remember, reflect, plan, and react believably over many simulated days?

## The Idea

A generative agent = an LLM + four architectural components layered around it:

1. **Memory stream** — an append-only, timestamped log of every observation and action, stored as natural-language sentences (e.g. "Isabella Rodriguez is setting up decorations at Hobbs Cafe at 2pm").
2. **Retrieval** — when the agent needs to decide what to do or say, don't dump the entire memory stream into the prompt (too long, mostly irrelevant). Instead score every memory and pull the top-scoring ones into context.
3. **Reflection** — periodically pause and synthesize higher-level thoughts from raw memories (e.g. many small observations about painting eventually produce the reflection "I am someone who is passionate about art").
4. **Planning & reacting** — generate a rough plan for the day, decompose it into finer-grained actions as the day unfolds, and revise the plan when something unexpected happens (a chance meeting, an invitation, an argument).

Concrete example from the paper: Isabella Rodriguez is initialized with the intent to plan a Valentine's Day party at Hobbs Cafe. She tells this to Maria Lopez, a customer, during a casual interaction. Maria mentions it to her friend Klaus Mueller while studying together. Over the simulated days that follow, without any of this being scripted beyond Isabella's initial intent, five agents learn about the party, two make plans to attend together as dates, and Isabella herself spends the day coordinating decorations and invitations — a chain of coherent, socially plausible cause and effect that spans the whole town's social graph.

## Architecture / Method

```
 perceive (see/hear something in the world)
        │
        ▼
 ┌───────────────────┐
 │   MEMORY STREAM    │  ← every observation/action appended, timestamped
 └─────────┬──────────┘
           │ score every memory by:
           │   recency   (exponential decay since last accessed)
           │   importance (LLM rates 1-10 "poignancy" at write time)
           │   relevance  (embedding similarity to current query)
           ▼
 ┌───────────────────┐
 │     RETRIEVE       │  → top-k memories surfaced into the prompt
 └─────────┬──────────┘
           │ periodically, when summed recent importance
           │ crosses a threshold:
           ▼
 ┌───────────────────┐
 │    REFLECTION      │  → LLM generates higher-level insights,
 └─────────┬──────────┘     citing which memories they came from
           │
           ▼
 ┌───────────────────┐
 │  PLAN & REACT       │  → coarse day-plan generated top-down,
 └─────────┬──────────┘     recursively decomposed into hourly/
           │                finer actions; re-planned on new,
           ▼                important observations
        act in the world
```

**Memory stream.** Every perception ("Isabella Rodriguez is setting up decorations") and every action taken is appended as a memory object with a timestamp. Nothing is deleted; the stream only grows.

**Retrieval.** Given a query (e.g. "what should I do right now"), every memory object is scored as a weighted combination of three signals — recency (an exponential decay function of time since the memory was last retrieved, so recently-touched memories stay salient), importance (a poignancy score from 1–10 that the LLM assigns when the memory is *written*, so "I broke up with my partner" scores much higher than "I brushed my teeth"), and relevance (cosine similarity between the query's embedding and the memory's embedding). The top-scoring memories (by the combined score) are pulled into the prompt context.

**Reflection.** Raw observations alone don't let an agent form opinions or notice patterns across many small events. When the sum of importance scores of recent memories crosses a threshold, the agent is prompted to (a) generate a small set of salient questions about itself/others given its recent memories, then (b) retrieve memories relevant to those questions and synthesize a higher-level statement, explicitly citing which underlying memories the insight is drawn from. Reflections are themselves stored back into the memory stream (as higher-poignancy memories), so reflections can build on earlier reflections — a recursive tree of increasingly abstract self-knowledge.

**Planning and reacting.** At the start of a simulated day, the agent generates a coarse plan (a handful of high-level activities across the day), then recursively decomposes each into finer sub-actions (hour blocks, then 5–15 minute chunks) only as needed, so compute isn't wasted planning far-future minute-level detail. When the agent perceives something that its current plan didn't anticipate, it decides whether the new information warrants reacting (interrupting the current plan) — and if so, may revise the remainder of the day's plan or trigger a dialogue with another agent.

## Key Results

- **Ablation study on believability.** Human evaluators rated agent believability under the full architecture vs. ablations that removed reflection, or removed planning, or used only recent-and-relevant retrieval without importance weighting. The full architecture was rated most believable; each ablation measurably degraded believability, showing all three components (retrieval scoring, reflection, planning) contribute independently.
- **Emergent diffusion of information.** The Valentine's Day party organized by Isabella Rodriguez was not scripted for any other agent — by the end of the simulation, of the 12 agents Isabella invited or who heard about it through word of mouth, several made and acted on plans to attend, entirely as a byproduct of individual agents' memory, retrieval, and planning loops interacting with each other.
- **Coherent multi-day behavior.** Agents maintained consistent identities, relationships, and daily routines (e.g. a character who runs a shop opens and closes it, restocks, and interacts with regular customers) over multiple simulated days, without a human re-prompting them each day.
- **New relationships and coordination.** Agents formed new relationships (e.g. two agents who hadn't previously interacted meeting, discovering a shared interest, and agreeing to meet again) without being explicitly scripted to.

## Comparison to Prior Work

- **vs. hand-authored NPC behavior (finite-state machines / behavior trees in games)** — those are believable only inside the exact situations their designer anticipated; they cannot improvise a response to a novel combination of events (a scheduling conflict, a rumor, an unplanned invitation). Generative agents improvise because the LLM generates open-ended responses conditioned on retrieved memory, not a fixed script.
- **vs. plain LLM-agent loops without structured memory** — an agent that's just re-prompted each turn with a short recent-history window forgets earlier interactions once they scroll out of context, and behaves incoherently over long horizons (contradicting itself, forgetting relationships, repeating actions). The memory stream + reflection + retrieval stack is precisely the fix for that forgetting problem.
- **vs. [[Reflexion]]** — Reflexion's verbal self-reflection is triggered by task failure and used to improve performance on a *retry* of the same task; Generative Agents' reflection is continuous and open-ended, synthesizing self-knowledge and social understanding across an unbounded, ongoing simulation rather than converging toward solving one task.

## Limitations

- **Compute cost.** Every agent, every simulated time step, involves multiple LLM calls (retrieval scoring context, reflection triggers, planning, dialogue generation) — this doesn't scale cheaply to hundreds or thousands of agents.
- **Hand-tuned heuristics.** The recency/importance/relevance weighting formula and the reflection-trigger threshold are manually chosen, not learned; different weightings could plausibly produce quite different (better or worse) behavior.
- **Toy sandbox evaluation.** Smallville is a small simulated town, not a real deployment — it's unclear how the architecture holds up in open-ended, adversarial, or much larger environments.
- **Long-horizon repetition/consistency.** The paper notes agents can still produce repetitive or occasionally inconsistent behavior over longer simulated periods than were tested.

## Why It Matters

This paper became the reference architecture for giving LLM agents long-term memory: the memory-stream + recency/importance/relevance retrieval + periodic reflection + hierarchical planning pattern (in adapted forms) shows up across later agent frameworks that need agents to behave coherently over more than a single prompt-response turn. It's a foundational citation for "LLM agents with long-term memory" and for social-simulation research more broadly.

## Related Concepts

[[Reflexion]] · [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] · [[In-Context Learning]] · [[World Models]] · [[LLM-Agent-Memory-Survey|Memory for Autonomous LLM Agents]] · [[Agentic-Memory-AgeMem|Agentic Memory (AgeMem)]]
