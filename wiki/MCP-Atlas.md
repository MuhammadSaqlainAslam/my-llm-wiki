---
created: "2026-06-03"
title: "MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers"
authors: "Chaithanya Bandi, Razvan-Gabriel Dumitru, Ben Hertzberg, Divyansh Agarwal, et al."
year: "2026"
arxiv: "2602.00933"
tags: [benchmarks, tool-use, mcp, agentic, evaluation, multi-step-reasoning, function-calling]
tldr: "1,000-task benchmark across 36 real MCP servers where the best model (Muse Spark) scores 82.2% pass rate; 63.3% of failures are cognitive rather than tool-call errors"
citation_count: 0
---

## TL;DR
MCP-Atlas is a large-scale benchmark measuring LLM agent tool-use competency against 36 production Model Context Protocol servers and 220 tools. It uses claim-level rubrics (not trajectory matching) so alternative valid tool paths get full credit. The key finding: most failures aren't because agents call the wrong tools — they're because agents fail to synthesize, stop too early, or misunderstand the task.

## The Problem
MCP has become the de facto standard for LLM-tool integration, but existing MCP evaluations fail on at least one of three axes:
1. **Realism**: Many use mocked APIs (MCPVerse, MCP-RADAR) rather than live servers with real rate limits, pagination, and errors
2. **Scale**: Rigorous benchmarks with programmatic verifiers top out at ~250 tasks — too small for statistically robust comparisons
3. **Scoring objectivity**: Holistic LLM-as-judge scoring (MCP-Bench, LiveMCPBench) introduces style bias — verbose correct answers may score differently than terse correct answers

The "unknown-tools" challenge is also underaddressed: most prior benchmarks give agents a clean, pre-filtered tool set rather than requiring discovery among semantic distractors.

## What It Measures

**Scale:**
- 1,000 tasks total (500 public / 500 private held-out split)
- 36 production MCP servers (containerized, version-pinned, sandboxed)
- 220 tools across 5 application domains
- 98.6% of tasks require cross-server orchestration

**Task structure per example:**
- 6–37 tools exposed per task (mean: 15.2 tools)
- Only 2–8 are relevant (mean: 4.1 required tools)
- Remaining tools are semantically plausible distractors — agent must discover which tools matter
- Prompts never name specific servers, tools, or parameters
- Multi-hop dependencies: later calls parameterized by earlier outputs

**Scoring:**
- **Claim-level rubric**: each task decomposed into atomic factual claims grounded in tool outputs
- Pass threshold: ≥0.75 claim coverage = pass (partial credit supported)
- Answer-centric, not trajectory-centric — valid alternative paths receive full credit
- 20 frontier models from 6 providers evaluated under matched task-level conditions

**Diagnostic taxonomy (11 categories):**
- Tool-call failures (36.7%): malformed parameters, wrong tool selection, no tool use, failed error recovery
- Cognitive failures (63.3%): task misunderstanding, faulty synthesis, misparsing, early termination, hallucinated facts, logical errors, constraint violations

## Key Results

| Model | Pass Rate (≥0.75 coverage) | Mean Coverage | Tier |
|-------|---------------------------|---------------|------|
| Muse Spark | 82.2% | 86.2% | Top |
| Claude Opus 4.7 | 79.1% | 83.8% | Top |
| Gemini 3.1 Pro Preview | 78.2% | 83.9% | Top |
| Claude Opus 4.6 | 76.8% | 82.7% | Mid |
| GLM-5.1 (open-source) | 75.6% | 81.8% | Mid |
| GPT-5.5 | 75.3% | 82.1% | Mid |
| GPT-5.4 | 70.6% | 78.5% | Mid |
| Gemini 3 Pro Preview | 70.3% | 79.0% | Mid |
| Claude Opus 4.5 | 69.8% | 76.9% | Mid |
| Claude Sonnet 4.6 | 69.5% | 78.0% | Mid |
| GPT-5.2 | 67.6% | 75.5% | Mid |
| Kimi K2.5 | 64.4% | 72.5% | Tail |
| Gemini 3 Flash Preview | 62.0% | 72.0% | Tail |
| Claude Sonnet 4.5 | 59.5% | 69.7% | Tail |
| GLM-4.7 | 58.1% | 68.5% | Tail |
| Gemini 3.1 Flash Lite Preview | 57.1% | 68.6% | Tail |
| GPT-5.4 Mini | 56.7% | 68.2% | Tail |
| GPT-5.1 | 50.1% | 61.5% | Tail |
| o3 Pro | 44.5% | 54.2% | Tail |
| Claude Haiku 4.5 | 40.2% | 51.7% | Tail |

**Key anomaly**: o3 Pro (44.5%) — a top performer on math/coding benchmarks — produces *no tool calls* on 40% of its failed tasks. Strong reasoning ability does not transfer to tool-use compliance.

**Open-source highlight**: GLM-5.1 (75.6%) enters the top empirical band, previously exclusive to proprietary models.

## Why It Matters for LLM Evaluation

- **Cognitive vs. tool failure decomposition**: 63.3% of diagnosed failures are cognitive (synthesis errors, premature stopping, hallucination) rather than tool mechanics — this reframes where model improvement effort should focus
- **Real servers, real failures**: All 36 servers are production implementations exposing authentic rate limits, pagination, schema mismatches, and transient errors — no mocks means no free passes on real-world edge cases
- **Distractor design reveals tool discovery gap**: With 6–37 tools per task and only 2–8 relevant, agents must perform genuine tool discovery among semantically plausible distractors — a harder and more realistic condition than prior benchmarks
- **Trajectory-agnostic scoring**: Claim-level rubrics decouple success from any single execution path, eliminating the bias where only the "expected" tool sequence gets credit
- **Scale enables statistical robustness**: At 1,000 tasks across 5 domains, performance gaps are statistically meaningful; prior MCP benchmarks with <250 tasks make model comparisons unreliable
- **Reasoning model policy mismatch exposed**: o3 Pro's collapse (44.5%) reveals that RLHF/reasoning training can actually suppress tool-calling behavior — a critical alignment failure mode for agentic deployment

## Limitations

- **Contamination risk**: 500-task public split is available; models trained after release could overfit, though the 500-task private split mitigates leaderboard gaming
- **Domain coverage**: 36 servers across 5 domains is broad but not exhaustive — enterprise, healthcare, and domain-specific APIs are likely underrepresented
- **Single-turn evaluation**: Tasks are framed as single-turn prompts despite requiring multi-step execution; true multi-turn interactive agent scenarios (e.g., clarification loops with users) are not captured
- **Claim authoring subjectivity**: Atomic claims are human-authored and verified, but the claim decomposition itself introduces judgment calls that could affect reproducibility across task authors
- **English-only**: No multilingual coverage evaluated
- **Dynamic servers**: Live endpoints introduce non-determinism (rate limits, transient errors) that could cause variance in scores across evaluation runs, even with containerization and version pinning
- **Scoring threshold sensitivity**: The 0.75 coverage threshold for "pass" is a design choice — results would look different at 0.5 or 0.9, and the threshold's calibration isn't extensively justified

## Related Concepts

[[LLM Benchmarks]] [[LLM evaluation]] [[RLHF]] [[Transformer]]

**Related benchmarks**: MCP-Universe, MCPEval, MCP-Bench, Toolathlon, MCPMark, LiveMCPBench, MCPVerse, MCP-RADAR, ToolBench, BFCL, WebArena, SWE-Bench, GAIA, MMLU, HELM

**Capabilities tested**: multi-step tool orchestration, tool discovery, cross-server reasoning, parameter synthesis, error recovery, long-horizon planning
