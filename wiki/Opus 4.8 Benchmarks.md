---
created: "2026-06-03"
title: "Opus 4.8 Benchmarks"
authors: "Anthropic"
year: "2026"
arxiv: ""
tags: [benchmarks, evaluation, claude, llm-comparison, agentic]
tldr: "Claude Opus 4.8 benchmark results vs Opus 4.7, GPT-5.5, and Gemini 3.1 Pro across agentic coding, reasoning, computer use, and tool use tasks"
citation_count: 0
---

## TL;DR
Claude Opus 4.8 (released May 28, 2026) leads on most agentic benchmarks — SWE-bench Pro (69.2%), OSWorld-Verified (83.4%), GDPval-AA (1890 Elo) — while GPT-5.5 edges it on Terminal-Bench 2.1 (78.2% vs 74.6%). Pricing unchanged at $5/$25 per 1M tokens. 4× improvement in code honesty (unflagged code flaws).

## Benchmark Results

### Full Comparison Table

| Benchmark | Opus 4.8 | Opus 4.7 | GPT-5.5 | Gemini 3.1 Pro |
|-----------|----------|----------|---------|----------------|
| **SWE-Bench Pro** (Pass@1) | **69.2%** | 64.3% | 58.6% | 54.2% |
| **Terminal-Bench 2.1** (Pass@1) | 74.6% | 66.1% | **78.2%** | 70.3% |
| **HLE** — no tools (Pass@1) | **49.8%** | 46.9% | 41.4% | 44.4% |
| **HLE** — with tools (Pass@1) | **57.9%** | 54.7% | 52.2% | 51.4% |
| **OSWorld-Verified** (Pass@1) | **83.4%** | 82.8% | 78.7% | 76.2% |
| **GDPval-AA** (Elo) | **1890** | 1753 | 1769 | 1314 |
| **Finance Agent v2** (Pass@1) | **53.9%** | 51.5% | 51.8% | 43.0% |

*Bold = best score in row.*

## What Each Benchmark Measures

- **[[SWE-bench Pro]]** — Hard real-world GitHub issue resolution, multi-file enterprise codebases, less contamination than the Verified variant
- **Terminal-Bench 2.1** — Linux shell agent tasks, bash commands, deterministic scoring; moved to v2.1 (not directly comparable to 4.7 era results)
- **[[Humanity's Last Exam]] (HLE)** — 2,500 expert-level academic questions across math, science, humanities — designed to resist saturation
- **[[OSWorld]]-Verified** — GUI agent controlling real desktop OS (Ubuntu/Windows/macOS) via mouse/keyboard; uses updated harness in 4.8
- **GDPval-AA** — Real economic value tasks, Elo-style rating by Artificial Analysis — 1890 Elo, 121 points ahead of GPT-5.5
- **Finance Agent v2** — Financial analysis agentic tasks

## Key Takeaways

- Opus 4.8 leads on 5 of 6 benchmarks
- GPT-5.5 wins Terminal-Bench 2.1 (78.2% vs 74.6%)
- **Biggest gap:** GDPval-AA — 121 Elo points ahead of GPT-5.5
- **Most significant non-benchmark change:** 4× reduction in unflagged code flaws (honesty improvement)
- **Tool-use uplift:** HLE with tools 57.9% vs without tools 49.8% — +8.1 points shows strong tool augmentation
- SWE-bench Pro 10-point lead over GPT-5.5 is the most contamination-resistant signal

## Why It Matters

- SWE-bench Pro is the most contamination-resistant coding benchmark available in 2026
- GDPval-AA measures real economic value delivered — hardest to game via training
- HLE with tools represents the frontier of tool-augmented reasoning
- Finance Agent v2 demonstrates domain-specific agentic capability

## Limitations

- OSWorld harness changed between versions — direct comparison to prior releases is not clean
- Terminal-Bench moved to v2.1 — not directly comparable to Opus 4.7 era results
- No vision-specific benchmarks published for Opus 4.8
- Self-reported by Anthropic — independent third-party verification pending
- GDPval-AA uses Elo, which is relative to other entrants at time of evaluation

## Related Papers

- [[SWE-bench]] — original coding benchmark (arXiv:2310.06770)
- [[SWE-bench Pro]] — harder enterprise variant (arXiv:2509.16941)
- [[OSWorld]] — computer use benchmark (arXiv:2404.07972)
- [[Humanity's Last Exam]] — expert reasoning benchmark (arXiv:2501.14249)
- [[MCP-Atlas]] — tool use with MCP servers (arXiv:2602.00933)
- [[LLM Benchmarks]] — broader frontier model comparison including Opus 4.6

## Data Quality Note

> Scores sourced from Anthropic official announcement (May 2026).
> Terminal-Bench 2.1 uses different harnesses across models (Codex CLI for GPT-5.5; Terminus-2 for others) — not directly comparable.
> OSWorld harness was updated between Opus 4.6 and 4.8 — part of the OSWorld gain is methodology.
> See [[Data Quality Standards]] for details on discrepancies and verification sources.

## Related Concepts

[[LLM Benchmarks]] · [[Data Quality Standards]] · [[LLM evaluation]] · [[SWE-bench]] · [[OSWorld]] · [[Humanity's Last Exam]] · [[MCP-Atlas]] · [[RLHF]] · [[RLVR]]
