---
created: "2026-10-19"
title: "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"
authors: "Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su (OSU / collaborators)"
year: 2024
arxiv: "2405.14831"
citation_count: 0
tags: [rag, retrieval, memory, knowledge-graph, multi-hop, long-context, agentic]
tldr: "Hippocampal-indexing-style RAG: LLM builds a KG (neocortex-ish), then Personalized PageRank over that graph (hippocampus-ish) does multi-hop retrieval in one step. Beats SOTA multi-hop RAG by up to 20%; matches IRCoT quality at 10–30× cheaper and 6–13× faster. NeurIPS 2024."
aliases: [HippoRAG, hippocampal RAG, long-term memory RAG]
theme: retrieval
---

# HippoRAG

> Bernal Jiménez Gutiérrez, et al., "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models", NeurIPS 2024 (arXiv:2405.14831)

## The Problem / Motivation

LLMs + vanilla [[RAG]] still **bad at integrating many new experiences** after pretraining. Multi-hop questions need several retrieval rounds (e.g. IRCoT-style loops) — expensive and brittle.

Mammals do not re-ask the world step by step every time: the **hippocampus indexes** patterns so neocortex-like knowledge can be linked and recalled quickly (hippocampal indexing theory).

## The Idea

Split memory roles and stitch them with **one graph walk**:

1. **Neocortex-like** — LLM extracts entities/relations into a **knowledge graph** over the corpus (offline index).
2. **Hippocampus-like** — **Personalized PageRank** over that KG from query anchors → multi-hop evidence without iterative LLM retrieval loops.
3. Answer with the retrieved passages.

**Single-step** retrieval that *spreads* activation over related entities — the “integrate experiences” move vanilla vector RAG lacks.

## Architecture / Method

```
docs ──LLM extract──► knowledge graph (nodes = entities/passages)
query ──anchors──► Personalized PageRank on KG
                 ──► ranked passages (multi-hop for free)
                 ──► LLM answer
```

- **Index once** (graph construction cost paid up front).
- **Query-time:** cheap PageRank-style spreading activation (not k sequential LLM retrievals).
- Can **stack** with iterative methods (HippoRAG + IRCoT) for extra gains.

## Key Results

| Axis | Result |
|---|---|
| Multi-hop QA | Up to **+20%** over prior SOTA RAG |
| vs iterative IRCoT | **Comparable or better** quality in **one** step |
| Cost / latency vs IRCoT | **10–30× cheaper**, **6–13× faster** |
| + IRCoT | Further substantial gains |
| New scenarios | Multi-hop / integration cases prior RAG cannot do (paper) |

## Comparison to Prior Work

- vs **[[RAG]] / vector retrieval** — single nearest chunks fail multi-hop; HippoRAG **links** entities first.
- vs **[[GraphRAG]]** — GraphRAG is map-reduce **global summarization** over communities; HippoRAG is **multi-hop retrieval** via PageRank indexing (orthogonal strengths).
- vs **[[RAPTOR]] / [[Self-RAG]]** — tree summaries / self-reflection on retrieved chunks; HippoRAG changes the **index structure**.
- vs **[[Agentic RAG Survey|Agentic RAG]]** — multi-step tool loops; HippoRAG gets hops from **graph structure** in one retrieve.

## Limitations

- Graph extraction quality (LLM) bounds recall — bad entities → bad PageRank.
- Index build cost and update cost when the corpus changes.
- Shines on multi-hop / relational Qs; simple fact lookup can be overkill vs vector RAG.
- KG is still a simplified model of “memory” (the neuro story is metaphor, not circuit-level).
- Eval is QA-centric; open-ended chat memory is a different problem ([LLM-Agent-Memory-Survey](LLM-Agent-Memory-Survey)).

## Why It Matters

This wiki’s memory/RAG line asks *how models hold and connect new knowledge*. HippoRAG is the **neuro-inspired multi-hop RAG** reference (NeurIPS 2024) next to [[GraphRAG]] (global) and [[RETRO]] (pretrained retrieval). The systems lesson is old and right: **structure the index so the hard search is not another LLM call**.

## Related Concepts

[[RAG]] · [[GraphRAG]] · [[Self-RAG]] · [[RAPTOR]] · [[RETRO]] · [[Lost in the Middle]] · [[Agentic RAG Survey]] · [[LLM-Agent-Memory-Survey]] · [[Agentic-Memory-AgeMem|Agentic Memory]] · [[Titans]] · [[Memory hierarchy]]