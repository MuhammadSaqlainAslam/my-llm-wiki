---
created: "2026-09-01"
title: "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"
authors: "Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei, Athanasios V. Vasilakos"
year: 2025
arxiv: "2501.09136"
tags: [retrieval, rag, agents, survey, agentic-workflows, multi-agent]
citation_count: 405
tldr: "Classic RAG retrieves once, generates once — a static workflow that can't adapt when the first retrieval misses, or when a task needs multiple reasoning steps. Agentic RAG embeds autonomous agents (reflection, planning, tool use, multi-agent collaboration) directly into the pipeline so retrieval strategy is decided dynamically. Proposes a taxonomy along agent cardinality, control structure, autonomy, and knowledge representation to organize an otherwise fragmented field."
aliases: ["Agentic RAG", "Agentic RAG Survey"]
---

# Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG

> Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei, Athanasios V. Vasilakos (Cleveland State University, Kent State University, The MathWorks Inc., Northeastern University), "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG", January 2025, revised April 2026 (arXiv:2501.09136)

## The Core Idea

The original [[RAG]] architecture already documented in this wiki does one retrieve-then-generate pass: encode the query, pull the top-$k$ passages, condition the generator on them, done. That's a huge upgrade over a purely parametric model, but it's still a **static workflow** — it can't notice that the first retrieval missed the point, can't decide to search again with a refined query, can't break a complex question into sub-questions and retrieve for each separately, and can't call a second specialized system when the retrieved text alone isn't enough.

Agentic RAG's core move is embedding autonomous agent behavior — the same design patterns this wiki tracks in [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]], [[Reflexion]], and [[Generative-Agents|Generative Agents]] — directly into the retrieval-augmented pipeline. Instead of one fixed pass, the system can reflect on whether its retrieved context is sufficient, plan a multi-step retrieval strategy, call external tools alongside the retriever, and coordinate multiple specialized agents on different sub-parts of a task. Retrieval becomes something the system *does*, iteratively and adaptively, rather than something that happens once at the start.

## Key Concepts / Taxonomy

The survey's central contribution is a taxonomy organizing the (previously fragmented, inconsistently-named) space of Agentic RAG systems along four dimensions:

- **Agent cardinality** — how many agents are involved: a single agent orchestrating its own retrieval and reasoning, or multiple specialized agents collaborating (a retrieval agent, a critique agent, a synthesis agent, etc.).
- **Control structure** — how the workflow is organized: a fixed pipeline with agentic steps inserted, a fully dynamic loop where the agent decides its own next action, or a hierarchical structure where a planner agent delegates to sub-agents.
- **Autonomy** — how much of the retrieval strategy (what to retrieve, when, whether to retry) is decided by the system itself versus fixed in advance by the designer.
- **Knowledge representation** — how retrieved and intermediate information is represented and passed between steps (raw passages, summarized intermediate state, structured memory).

The agentic design patterns that recur across the taxonomy: **reflection** (assess whether current retrieved context/answer is good enough), **planning** (decompose a complex query into a retrieval strategy), **tool use** (call external systems beyond the retriever), and **multi-agent collaboration** (specialized agents handling different sub-tasks).

## Architecture / Method

```
Classic RAG (static, single-pass)          Agentic RAG (dynamic, iterative)
──────────────────────────────             ─────────────────────────────────
   query                                        query
     │                                            │
     ▼                                            ▼
  retrieve top-k                            ┌─────────────────┐
     │                                      │  Agent decides:   │
     ▼                                      │  - retrieve again?│
  generate                                  │  - refine query?  │
     │                                      │  - call a tool?   │
     ▼                                      │  - delegate to    │
   answer                                   │    another agent? │
                                             └─────────────────┘
                                                    │  loop until
                                                    │  agent judges
                                                    │  answer sufficient
                                                    ▼
                                                  answer
```

Concretely, an Agentic RAG system might: retrieve, reflect on whether the retrieved passages actually answer the question (a pattern shared with [[Self-RAG]]'s ISREL/ISSUP reflection tokens, though Self-RAG bakes reflection into token-level training rather than an external agent loop), retrieve again with a refined query if not, plan a decomposition of a multi-hop question into sub-questions each requiring their own retrieval (echoing [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]]'s interleaved Thought/Act/Observation loop, but applied specifically to retrieval as the action), or hand off to a specialized sub-agent for a domain the main agent isn't equipped to handle.

## Comparison to Prior Work

- vs. **[[RAG]]** (classic retrieve-then-generate) — Agentic RAG is explicitly framed as the next evolution beyond RAG's static workflow; where RAG treats retrieval as a fixed pre-processing step, Agentic RAG treats it as a decision the system makes repeatedly and adaptively.
- vs. **[[Self-RAG]]** — Self-RAG achieves adaptivity (deciding when to retrieve, and critiquing its own output) by training the base model to emit special reflection tokens; it's "agentic" behavior baked into a single model's generation process rather than an external multi-step agent loop. The taxonomy in this survey would classify Self-RAG as low agent-cardinality (one model) with high autonomy but a narrower control structure than a full multi-agent Agentic RAG system.
- vs. **[[RAPTOR]]** — RAPTOR improves *what gets retrieved* (a recursive summarization tree instead of flat passages) but the retrieval process itself is still a single pass; it's a knowledge-representation improvement, not an agentic-control-structure one, in this survey's terms.
- vs. **[[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] and [[Reflexion]]** — those papers established the general reasoning-plus-acting and verbal-self-reflection patterns for LLM agents; Agentic RAG is what those patterns look like when the "acting" is specifically retrieval, and the survey's taxonomy is a way of cataloging the many systems that have since applied this combination.

## Limitations

- As a survey, it doesn't propose or evaluate a new system — its contribution is organizational (the taxonomy) plus a synthesis of applications and open challenges, not a benchmark result.
- The field it's surveying is fragmented and fast-moving (the paper itself has been revised four times as of April 2026 to keep pace), so the taxonomy is necessarily a snapshot rather than a stable final classification.
- Open challenges the survey identifies — evaluation, coordination, memory management, efficiency, and governance — are flagged but not solved; a reader looking for concrete answers to "how do I evaluate an Agentic RAG system" will find the problem named, not resolved.

## Why It Matters

This survey sits at the intersection of the two newest theme sections in this wiki, [[RAG]] and this wiki's LLM-agent notes ([[Generative-Agents|Generative Agents]], [[Reflexion]]) — it's the paper that names and organizes what happens when those two lines of work combine, which is exactly where most production "AI assistant with document search and tool access" systems actually live today. Its taxonomy (agent cardinality, control structure, autonomy, knowledge representation) gives a shared vocabulary for comparing systems that otherwise get described with inconsistent, marketing-driven terminology ("agentic," "autonomous," "multi-step RAG" all get used loosely) — useful for anyone trying to figure out what a given production RAG system is actually doing under the hood.

## Related Notes

[[RAG]] · [[Self-RAG]] · [[RAPTOR]] · [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] · [[Reflexion]] · [[Generative-Agents|Generative Agents]]
