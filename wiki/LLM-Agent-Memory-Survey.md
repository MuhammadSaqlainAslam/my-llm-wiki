---
created: "2026-09-01"
title: "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers"
authors: "Pengfei Du"
year: 2026
arxiv: "2603.07670"
tags: [agents, memory, survey, retrieval, reflection, taxonomy]
citation_count: 55
tldr: "Formalizes agent memory as a write-manage-read loop coupled to perception and action, with a taxonomy spanning temporal scope, representational substrate, and control policy. Surveys five mechanism families — context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, and policy-learned management — and the applications where memory is the differentiator: personal assistants, coding agents, open-world games, scientific reasoning, multi-agent teamwork."
aliases: ["LLM Agent Memory Survey", "Memory for Autonomous LLM Agents"]
---

# Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers

> Pengfei Du (Hong Kong Research Institute of Technology), "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers", March 2026 (arXiv:2603.07670)

## The Core Idea — The Write-Manage-Read Loop

A single context window is far too small to capture everything an agent has done, learned, and should not repeat across a long-running task. Memory is what separates a stateless text generator from a genuinely adaptive agent — this is the same gap [[Generative-Agents|Generative Agents]] and [[Reflexion]] each solve in their own way, and this survey's job is to formalize what "agent memory" actually means across all such systems.

The proposed formalization: agent memory is a **write → manage → read loop**, tightly coupled with the agent's perception (what it observes) and action (what it does next):

- **Write** — deciding what's worth storing from the current experience.
- **Manage** — deciding what to keep, compress, forget, or promote to longer-term storage as memory accumulates.
- **Read** — deciding what to retrieve from memory and how to inject it back into the current context when acting.

Every specific memory architecture in the literature — including the ones already documented in this wiki — is an instantiation of a policy for these three operations.

## Key Concepts / Taxonomy

The survey's three-dimensional taxonomy for classifying any given memory mechanism:

- **Temporal scope** — how long information persists: within a single episode/task, across sessions, or permanently.
- **Representational substrate** — the form memory takes: raw text logs, embeddings in a vector store, structured/symbolic summaries, or model weights themselves (memory via fine-tuning).
- **Control policy** — how write/manage/read decisions are made: fixed heuristics, learned/trained policies, or the LLM's own judgment at inference time.

Five mechanism families examined in depth:

1. **Context-resident compression** — summarize or compress history so it fits back into the active context window, rather than storing it externally.
2. **Retrieval-augmented stores** — write experience into an external store (often vector-indexed) and read back the most relevant entries at each step — the family [[Generative-Agents|Generative Agents]]' memory stream belongs to.
3. **Reflective self-improvement** — periodically generate higher-level reflections or lessons from raw experience, and store those reflections rather than (or in addition to) raw logs — the family [[Reflexion]]'s verbal self-reflection buffer belongs to.
4. **Hierarchical virtual context** — organize memory in layers of increasing abstraction (recent detail, medium-term summary, long-term gist), read from the appropriate layer depending on the task.
5. **Policy-learned management** — train the write/manage/read decisions themselves (e.g., via reinforcement learning) rather than fixing them as heuristics.

## Architecture / Method

```
   Perception (observe)         Action (act)
        │                            ▲
        ▼                            │
   ┌─────────────────────────────────────┐
   │              AGENT                    │
   │                                        │
   │   WRITE ──────▶ MANAGE ──────▶ READ    │
   │   (what to      (compress/     (what to │
   │    store)        forget/        retrieve │
   │                  promote)       & inject) │
   └─────────────────────────────────────┘
        │
        ▼
   Memory store — one of five mechanism families:
   context-compression | retrieval store | reflection buffer |
   hierarchical virtual context | policy-learned management
```

Applications where the survey argues memory is the differentiating factor rather than a peripheral feature: personal assistants (remembering user preferences across sessions), coding agents (remembering prior file edits and failed approaches within a long task), open-world games (remembering world state and past interactions — directly overlapping with [[Generative-Agents|Generative Agents]]' Smallville setting), scientific reasoning (remembering intermediate derivations across a long multi-step proof or experiment), and multi-agent teamwork (shared or coordinated memory across cooperating agents).

## Comparison to Prior Work

- vs. **[[Generative-Agents|Generative Agents]]** — its memory stream (recency + importance + relevance retrieval, plus periodic reflection) is a concrete instance of this survey's taxonomy: primarily a **retrieval-augmented store** (family 2) with an added **reflective self-improvement** layer (family 3), using a heuristic (not learned) control policy.
- vs. **[[Reflexion]]** — its Actor/Evaluator/Self-Reflection loop is almost entirely a **reflective self-improvement** mechanism (family 3): no external vector store, no gradient updates — the "memory" is a growing buffer of verbal self-critiques fed back into the next attempt's context, with temporal scope limited to within a single task's retry loop rather than persisting indefinitely.
- vs. **[[RAG]] / [[RAPTOR]]** — these are retrieval systems over a fixed external corpus (documents), not memory of the *agent's own experience*; the survey's taxonomy is specifically about the latter, though the retrieval mechanics (dense vector search, hierarchical summarization trees) directly transfer, and RAPTOR's recursive-tree structure is a natural fit for this survey's "hierarchical virtual context" family.

## Limitations

- Being a synthesis survey, it doesn't introduce a new memory mechanism or benchmark of its own — its contribution is the organizing taxonomy and the write-manage-read framing, not new empirical results.
- The five mechanism families aren't mutually exclusive in practice (most real systems, including [[Generative-Agents|Generative Agents]], combine two or more), so classifying any given system can require judgment calls about which family is "primary."
- Evaluation methodology for agent memory specifically is flagged as an open frontier by the survey itself — there's no single agreed benchmark for "how good is this agent's memory," which limits how precisely different mechanism families can be compared.

## Why It Matters

This survey gives a shared vocabulary for a component that nearly every serious LLM agent system needs but that this wiki's existing notes ([[Generative-Agents|Generative Agents]], [[Reflexion]]) each describe in their own bespoke terms. The write-manage-read framing is a useful lens for reading *any* agent paper: instead of asking "does it have memory," ask what its write policy stores, how its manage policy decides what survives, and what its read policy retrieves — which immediately clarifies what's actually novel about a new memory-architecture paper versus what's a repackaging of an existing mechanism family.

## Related Notes

[[Generative-Agents|Generative Agents]] · [[Reflexion]] · [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] · [[RAG]] · [[RAPTOR]] · [[Agentic-Memory-AgeMem|Agentic Memory (AgeMem)]]
