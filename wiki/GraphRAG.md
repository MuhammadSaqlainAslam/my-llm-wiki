---
created: "2026-10-19"
title: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization (GraphRAG)"
authors: "Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson (Microsoft Research)"
year: 2024
arxiv: "2404.16130"
citation_count: 0
tags: [rag, retrieval, agents, summarization, knowledge-graph, long-context, agentic]
tldr: "Global questions ('main themes?') break chunk-level RAG. GraphRAG builds an entity KG from the corpus, clusters it into communities, pre-writes community summaries, then answers by map-reduce over those summaries. Beats conventional RAG on comprehensiveness and diversity for global sensemaking on ~1M-token corpora."
aliases: [GraphRAG, Graph RAG, query-focused summarization]
theme: retrieval
---

# GraphRAG

> Darren Edge, et al. (Microsoft Research), "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", 2024 (arXiv:2404.16130)

## The Problem / Motivation

Classic [[RAG]] (retrieve chunks → stuff into prompt) is built for **local** questions: *find the passage that answers this*. It collapses on **global** questions over a private corpus:

> “What are the main themes in the dataset?”  
> “What do people argue about?”  
> “Summarize the key risks across these contracts.”

Those are **query-focused summarization (QFS)** over the whole collection, not top-k retrieval. Naïve QFS does not scale to RAG-sized indexes.

## The Idea

**Index the corpus as a graph of entities + community summaries**, then answer global questions with **map-reduce over communities** (local → global).

1. **Build** — LLM extracts entities/relations → knowledge graph → hierarchical **communities** (Leiden-style clustering).
2. **Pre-generate** — a short **summary per community** (and optional community reports at multiple levels).
3. **Answer** — for a query: use each (relevant) community summary to write a partial answer → **summarize the partials** into the final reply.

Local QFS-like questions can also use community structure; the headline win is **global sensemaking**.

## Architecture / Method

```
documents
   │ LLM extract
   ▼
entity/relation KG  ──►  community detection (multi-level)
   │                         │
   │                         ▼
   │                  community summaries (cached)
   ▼                         │
graph context ◄──────────────┘
   │
query ──► map: partial answers per community ──► reduce: final summary
```

- **Graph index** is LLM-built (entities, claims, links) — not a plain vector store.
- **Communities** group densely related entities; summaries are the unit of “global context.”
- **Map-reduce generation** scales with #communities, not #tokens in one prompt.

## Key Results

| Setting | Result |
|---|---|
| Global / sensemaking Qs on ~1M token datasets | **Substantial gains** vs conventional RAG on **comprehensiveness** and **diversity** of answers (abstract) |
| Scope | Private / previously unseen corpora (the RAG setting) |
| Tradeoff | Heavier indexing (LLM extraction + community summaries) for better global answers |

## Comparison to Prior Work

- vs **[[RAG]] (vector chunks)** — GraphRAG answers *corpus-level* questions; vector RAG retrieves snippets.
- vs **[[Self-RAG]]** / **[[RAPTOR]]** — adaptive retrieval and tree summarization; GraphRAG’s unit is **entity community** + precomputed summary.
- vs **[[Lost in the Middle]]** — long-context placement issues; GraphRAG shrinks the prompt to summary set.
- vs **[[Agentic RAG Survey|Agentic RAG]]** — multi-step retrieval agents; GraphRAG is a structured index + map-reduce, often complementary.

## Limitations

- **Index cost** — many LLM calls for extraction and community reports; not free on huge corpora.
- Quality depends on entity extraction and clustering; messy domains → messy communities.
- Overkill for simple factoid QA where vector RAG is enough.
- “Comprehensiveness / diversity” wins are judge-style metrics; domain evals still matter.
- Graph + summaries can go stale when source docs change (reindex).

## Why It Matters

This wiki’s RAG/agent thread is about **what you feed the model** when the answer is not one chunk. GraphRAG is the canonical **global / corpus-level** RAG design (Microsoft) and the reference people name next to [[RAPTOR]] and [[Self-RAG]]. It matches the Karpathy-style moral here: *if the task is summarization over a corpus, retrieve/structure for that task* — don’t bolt top-k on and hope.

## Related Concepts

[[RAG]] · [[RAG]] · [[Self-RAG]] · [[RAPTOR]] · [[Lost in the Middle]] · [[Agentic RAG Survey]] · [[LLM-Agent-Memory-Survey]] · [[RETRO]] · [[Tree-of-Thoughts]] · [[SWE-bench]]