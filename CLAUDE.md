# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

An Obsidian-based knowledge wiki covering LLM architecture — written in the style of Andrej Karpathy (intuition first, math second, no fluff). All content lives in `wiki/` as interlinked Markdown; source academic PDFs are in `raw/`.

## Environment Setup

To activate the Conda environment and Azure Foundry integration before running Claude Code:

```bash
source activate_claude_w_Azure_API.sh
```

This activates the `saqlain_vllm` Conda environment and sets `CLAUDE_CODE_USE_FOUNDRY=1` with Azure endpoints.

## Content Structure

- `wiki/Home.md` — top-level index with concept map and recommended reading order
- `wiki/000 Karpathy Index.md` — full article list
- `wiki/*.md` — individual concept articles
- `raw/*.pdf` — source papers (Attention Is All You Need, Mamba, Mixtral, Switch Transformers, Nemotron-3)

## Architecture / Knowledge Graph

```
Transformer (2017)
│
├── O(n²) attention cost → Mamba: selective SSM, linear-time
├── Parameters tied to compute → Mixture-of-Experts: sparse routing
└── MoE communication bottlenecks → Nemotron-3 LatentMoE: route in latent space

Mamba + MoE + few attention layers = Nemotron-3 hybrid (best throughput-accuracy frontier, 2025)
```

Additional topics: KV Cache, GQA, Speculative Decoding, Multi-Token Prediction, State Space Models, RLVR, NVFP4, Hardware-Aware Scan, Load Balancing Loss.

## Writing Style for New Articles

Follow the pattern in existing articles: lead with intuition and motivation, use concrete analogies, introduce math only after the concept is clear. Use Obsidian `[[WikiLink]]` format for cross-references between articles.
