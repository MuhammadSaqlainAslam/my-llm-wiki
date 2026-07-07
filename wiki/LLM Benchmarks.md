---
created: "2026-05-05"
title: "LLM Benchmarks"
tags: [benchmarks, evaluation, llm-comparison, claude, gpt, gemini, agentic]
year: "2026"
tldr: "Verified benchmark scores for frontier LLMs across coding, reasoning, long-context, and agentic tasks. Sourced from official system cards and independent evaluators. Last updated June 2026."
citation_count: 0
---

# LLM Benchmarks

> ⚠️ All scores are vendor self-reported unless marked **(independent)**.
> Where vendor and independent sources disagree, both are shown.
> Last updated: June 2026.

---

## Claude Opus 4.8 (May 28, 2026)

Source: Anthropic official announcement (May 2026).
See [[Opus 4.8 Benchmarks]] for full analysis.

### Agentic & Coding

| Benchmark | Opus 4.8 | Opus 4.7 | GPT-5.5 | Gemini 3.1 Pro |
|-----------|----------|----------|---------|----------------|
| **SWE-Bench Pro** (Pass@1) | **69.2%** | 64.3% | 58.6% | 54.2% |
| **Terminal-Bench 2.1** (Pass@1) | 74.6% | 66.1% | **78.2%** | 70.3% |
| **OSWorld-Verified** (Pass@1) | **83.4%** | 82.8% | 78.7% | 76.2% |
| **Finance Agent v2** (Pass@1) | **53.9%** | 51.5% | 51.8% | 43.0% |

### Reasoning

| Benchmark | Opus 4.8 | Opus 4.7 | GPT-5.5 | Gemini 3.1 Pro |
|-----------|----------|----------|---------|----------------|
| **HLE** — no tools (Pass@1) | **49.8%** | 46.9% | 41.4% | 44.4% |
| **HLE** — with tools (Pass@1) | **57.9%** | 54.7% | 52.2% | 51.4% |
| **GDPval-AA** (Elo) **(independent)** | **1890** | 1753 | 1769 | 1314 |

**Notes:**
- Terminal-Bench 2.1: GPT-5.5 uses Codex CLI harness; other models use Terminus-2 harness — not directly comparable.
- OSWorld uses updated harness in 4.8 — part of the gain is methodology, not pure model improvement.
- 4× improvement in unflagged code flaws (internal honesty metric; not independently verified).

*Bold = best score in row.*

---

## Claude Opus 4.6 (February 5, 2026)

Source: Anthropic official system card (February 2026).

### Long Context

| Benchmark | Opus 4.6 | Sonnet 4.5 | GPT-5.2 | Gemini 3 Pro |
|-----------|----------|------------|---------|--------------|
| MRCR v2 1M 8-needle (MMR) | **76%** | 18.5% | 18.5% | 26.3% |
| MRCR v2 256k 8-needle (MMR) | **~77%** | — | — | 77% |

> **MRCR = Multi-Round Coreference Resolution.** 8-needle = hardest setting; 1M = prompts with (524k, 1024k] tokens. Score = Mean Match Ratio. Opus 4.6 leads all models as of June 2026 — Opus 4.7 regressed on this benchmark.

> ⚠️ **Score discrepancy:** The broader model comparison table below (from an internal image) shows Opus-4.6 MRCR 1M at **92.9**. The official system card shows **76%** (MRCR v2, 8-needle hardest variant). Likely causes: different MRCR variant (v1 vs v2), different needle count, or different context-length bin. **Use 76% when citing from the official system card.**

### Core Benchmarks

| Benchmark | Score | Notes |
|-----------|-------|-------|
| SWE-bench Verified (Resolved) | 80.84% | |
| SWE-bench Pro (Resolved) | 57.3% | |
| Terminal-Bench 2.0 (Acc) | 65.4% | State-of-the-art at release |
| OSWorld-Verified (Pass@1) | 72.7% | |
| ARC-AGI-2 (Pass@1) | 68.8% | |
| GPQA Diamond (Pass@1) | 91.3% | |
| HLE with tools (Pass@1) | 53.0% | Updated Feb 23 2026 after cheating-detection improvement |
| HLE without tools (Pass@1) | ~40.0% | |
| GDPval-AA (Elo) **(independent)** | 1,606 | 144 pts ahead of GPT-5.2 at evaluation date |
| BrowseComp (Pass@1) | SOTA | Leads all frontier models |

> ⚠️ **GDPval-AA discrepancy:** The broader comparison table below shows 1,619 Elo; the official system card shows 1,606. Difference of 13 points — likely different evaluation date or Elo pool. Use 1,606 from the official system card.

**Key facts:**
- Released February 5, 2026; pricing $5/$25 per 1M tokens.
- First Opus with 1M token context (beta, premium pricing above 200k tokens).
- Best long-context model of 2026 — Opus 4.7 regressed on MRCR v2.
- HLE with tools updated from 53.1% to 53.0% on February 23 2026 after cheating-detection pipeline improvement.

---

## Broader Model Comparison (early 2026)

Source: internal comparison image shared by HHRI-AI Research team.
⚠️ Primary source unknown for several scores — cross-check against official cards before citing.

### Models

| Short name | Full name | Provider |
|---|---|---|
| DS-V4-Pro | DeepSeek-V4-Pro Max | DeepSeek-AI |
| DS-V4-Flash | DeepSeek-V4-Flash Max | DeepSeek-AI |
| K2.6 | [[Kimi K2|K2.6 Thinking]] | Unknown / Kimi |
| GLM-5.1 | [[GLM-5|GLM-5.1 Thinking]] | Zhipu AI |
| Opus-4.6 | Claude Opus 4.6 Max | Anthropic |
| GPT-5.4 | GPT-5.4 xHigh | OpenAI |
| Gem-3.1-Pro | Gemini-3.1-Pro High | Google DeepMind |

### Knowledge & Reasoning

| Benchmark | DS-V4-Pro | DS-V4-Flash | K2.6 | GLM-5.1 | Opus-4.6 | GPT-5.4 | Gem-3.1-Pro |
|---|---|---|---|---|---|---|---|
| MMLU-Pro (EM) | 87.5 | 86.2 | 87.1 | 86.0 | 89.1 | 87.5 | **91.0** |
| SimpleQA-Verified (Pass@1) | 57.9 | 34.1 | 36.9 | 38.1 | 46.2 | 45.3 | **75.6** |
| Chinese-SimpleQA (Pass@1) | 84.4 | 78.9 | 75.9 | 75.0 | 76.2 | 76.8 | **85.9** |
| GPQA Diamond (Pass@1) | 90.1 | 88.1 | 90.5 | 86.2 | 91.3 | 93.0 | **94.3** |
| HLE (Pass@1) | 37.7 | 34.8 | 36.4 | 34.7 | 40.0 | 39.8 | **44.4** |
| LiveCodeBench (Pass@1) | **93.5** | 91.6 | 89.6 | — | 88.8 | — | 91.7 |
| Codeforces (Rating) | **3206** | 3052 | — | — | — | 3168 | 3052 |
| HMMT 2026 Feb (Pass@1) | 95.2 | 94.8 | 92.7 | 89.4 | 96.2 | **97.7** | 94.7 |
| IMOAnswerBench (Pass@1) | 89.8 | 88.4 | 86.0 | 83.8 | 75.3 | **91.4** | 81.0 |
| Apex (Pass@1) | 38.3 | 33.0 | 24.0 | 11.5 | 34.5 | 54.1 | **60.9** |
| Apex Shortlist (Pass@1) | **90.2** | 85.7 | 75.5 | 72.4 | 85.9 | 78.1 | 89.1 |

### Long Context

| Benchmark | DS-V4-Pro | DS-V4-Flash | K2.6 | GLM-5.1 | Opus-4.6 | GPT-5.4 | Gem-3.1-Pro |
|---|---|---|---|---|---|---|---|
| MRCR 1M (MMR) | 83.5 | 78.7 | — | — | 92.9† | — | 76.3 |
| CorpusQA 1M (ACC) | 62.0 | 60.5 | — | — | 71.7 | — | 53.8 |

> †Opus-4.6 MRCR 1M score of **92.9** is from this comparison image (source unknown). Official Anthropic system card reports **76%** using MRCR v2 8-needle. These likely measure different variants — see the discrepancy note in the Opus 4.6 section above.

### Agentic

| Benchmark | DS-V4-Pro | DS-V4-Flash | K2.6 | GLM-5.1 | Opus-4.6 | GPT-5.4 | Gem-3.1-Pro |
|---|---|---|---|---|---|---|---|
| Terminal Bench 2.0 (Acc) | 67.9 | 56.9 | 66.7 | 63.5 | 65.4 | **75.1** | 68.5 |
| SWE Verified (Resolved) | 80.6 | 79.0 | 80.2 | — | **80.8** | — | 80.6 |
| SWE Pro (Resolved) | 55.4 | 52.6 | **58.6** | 58.4 | 57.3 | 57.7 | 54.2 |
| SWE Multilingual (Resolved) | 76.2 | 73.3 | 76.7 | 73.3 | **77.5** | — | — |
| BrowseComp (Pass@1) | 83.4 | 73.2 | 83.2 | 79.3 | 83.7 | 82.7 | **85.9** |
| HLE w/tools (Pass@1) | 48.2 | 45.1 | **54.0** | 50.4 | 53.0† | 52.0 | 51.6 |
| GDPval-AA (Elo) **(independent)** | 1554 | 1395 | 1482 | 1535 | 1619† | **1674** | 1314 |
| MCPAtlas Public (Pass@1) | 73.6 | 69.0 | 66.6 | 71.6 | **73.8** | 67.2 | 69.2 |
| Toolathlon (Pass@1) | 51.8 | 47.8 | 50.0 | 40.7 | 47.2 | **54.6** | 48.8 |

> †HLE w/tools: image shows 53.1; official system card shows 53.0% (updated Feb 23 2026). †GDPval-AA: image shows 1,619 Elo; official system card shows 1,606. Both marked with † for cross-check.

*Bold = best score in row. — = not reported.*

---

## TL;DR

No single model wins everywhere. Gemini-3.1-Pro High dominates Knowledge & Reasoning. Opus-4.6 Max leads long-context (76% MRCR v2 8-needle per official system card). Agentic tasks are fragmented: GPT-5.4 leads GDPval-AA and Terminal Bench; [[Kimi K2|K2.6]] leads SWE Pro and HLE-with-tools; Opus-4.6 leads SWE Verified, SWE Multilingual, and MCPAtlas. DS-V4-Pro is the strongest open model. Opus 4.8 (May 2026) leads on SWE-bench Pro (69.2%), OSWorld (83.4%), GDPval-AA (1890 Elo), and HLE (57.9% with tools).

---

## Benchmark Descriptions

### [[SWE-bench]] / SWE-bench Pro
Real GitHub issue resolution. Pro uses harder enterprise codebases.
arXiv: [2310.06770](https://arxiv.org/abs/2310.06770) · [2509.16941](https://arxiv.org/abs/2509.16941)

### MRCR v2 (Multi-Round Coreference Resolution)
Tests retrieval of specific information buried in massive contexts.
8-needle = hardest variant; 1M = (524k, 1024k] token prompts. Score = Mean Match Ratio.

### [[Humanity's Last Exam]] (HLE)
2,500 expert-level questions across math, science, humanities.
arXiv: [2501.14249](https://arxiv.org/abs/2501.14249)

### [[OSWorld]]
GUI agent controlling real desktop OS (Ubuntu/Windows/macOS), 369 tasks.
arXiv: [2404.07972](https://arxiv.org/abs/2404.07972)

### GDPval-AA (independent)
Economically valuable knowledge-work tasks. Elo-style rating by Artificial Analysis.
Not vendor self-reported — comparisons here are more trustworthy.

---

## Sources

| Source | Type | Notes |
|--------|------|-------|
| Anthropic system card (Opus 4.6, Feb 2026) | Vendor | Official methodology; use for MRCR and GDPval |
| Anthropic announcement (Opus 4.8, May 2026) | Vendor | Official; harness notes apply |
| Internal comparison image (HHRI-AI team) | Internal | Source of broader 7-model table; primary origin unknown |
| Artificial Analysis GDPval-AA | Independent | Most trustworthy for Elo comparisons |
| HLE leaderboard (agi.safe.ai) | Independent | Verified HLE scores |
| SWE-bench leaderboard (swebench.com) | Independent | Verified SWE scores |

---

## Related Concepts

[[Opus 4.8 Benchmarks]] · [[SWE-bench]] · [[OSWorld]] · [[Humanity's Last Exam]] · [[MCP-Atlas]] · [[DeepSeek_V4]] · [[KV cache]] · [[Speculative Decoding]] · [[GQA]] · [[RLVR]] · [[LLM evaluation]]
