---
created: "2026-06-03"
title: "OSWorld-MCP: Evaluating Agents with MCP Tool Invocation in Computer Environments"
authors: "Hongrui Jia, Jitong Liao, Xi Zhang, Haiyang Xu, Tianbao Xie, Chaoya Jiang, Ming Yan, Si Liu, Wei Ye, Fei Huang"
year: "2025"
arxiv: "2510.24563"
tags: [benchmarks, tool-use, mcp, computer-use, evaluation, multimodal-agents, gui]
tldr: "OSWorld-MCP extends OSWorld with 158 MCP tools across 7 apps, revealing that tool access improves success rates dramatically (8.3%→20.4% for o3) while best models still only invoke tools 36.3% of the time."
citation_count: 13
---

## TL;DR
OSWorld-MCP is the first benchmark to jointly evaluate computer-use agents on GUI operations, MCP tool invocation, and decision-making in a unified real-world environment. It adds 158 curated MCP tools to the existing OSWorld framework, covering 7 applications and 250 applicable tasks (69% of the benchmark). The core finding: tools help a lot, but current models are bad at actually using them.

## The Problem
Existing GUI agent benchmarks (OSWorld, WebArena, WindowsAgentArena) only allow predefined GUI actions — click, type, drag. Meanwhile, several recent agents (e.g., from Lai et al., Song et al.) have started shipping with MCP tool invocation baked in and achieving notable gains. Comparing these tool-augmented agents against pure-GUI baselines on GUI-only benchmarks is **fundamentally unfair** — it's like scoring a calculator user on arithmetic the same way you score a mental math contestant.

The gap: no benchmark simultaneously tests (1) GUI interaction skill, (2) tool invocation capability, and (3) the *decision* of which approach to use when both are available.

## What It Measures

**Benchmark structure:**
- Built on top of **OSWorld** (real Ubuntu desktop environment)
- **158 high-quality MCP tools** covering 7 applications: LibreOffice Writer, VS Code, and others
- **25 non-target distractor tools** included to test selection precision
- **~362 total tasks** (250 tool-applicable = 69%; 153 require multi-round tool chaining = 42%)
- Agents tested at **15-step and 50-step** horizons

**Task complexity tiers:**
- Single-tool tasks (simpler)
- Multi-round tool invocation (up to 4 rounds; Claude 4 Sonnet scores **0% GUI-only**, 16.7% with tools at 4-round tasks)

**Metrics:**
| Metric | Description |
|--------|-------------|
| Task Success Rate | Binary: task completed correctly |
| Tool Invocation Rate (TIR) | % of tasks where agent successfully uses MCP tools |
| Average Completion Steps (ACS) | Efficiency: how many steps to complete a task |

**Tool creation pipeline:**
- Automated code generation (o3-powered) → Code Filter → Tool Wrap → manual curation
- 72 auto-generated tools + curated from existing MCP servers → 158 final tools after deduplication and quality filtering

**Scoring:** Pass@1 style binary success evaluated in live interactive environment (not static replay).

## Key Results

| Model | Setting | Steps | Success Rate | Tool Invocation Rate |
|-------|---------|-------|-------------|---------------------|
| Claude 4 Sonnet | GUI only | 50 | 40.1% | — |
| Claude 4 Sonnet | MCP tools | 50 | 43.3% | ~36.3% (best) |
| OpenAI o3 | GUI only | 15 | 8.3% | — |
| OpenAI o3 | MCP tools | 15 | 20.4% | — |
| Claude 4 Sonnet | GUI only, 4-round tasks | 50 | 0% | — |
| Claude 4 Sonnet | MCP tools, 4-round tasks | 50 | 16.7% | — |

**Key numbers to internalize:**
- Best TIR achieved: **36.3%** — even the top model ignores available tools ~64% of the time
- o3 improvement with tools: **+12.1 pp** (8.3% → 20.4%) at 15 steps
- Claude 4 Sonnet improvement: **+3.2 pp** (40.1% → 43.3%) at 50 steps (more steps = more GUI recovery possible)

## Why It Matters for LLM Evaluation

- **Exposes a hidden capability gap**: TIR of only 36.3% for the best model means current LMMs fundamentally don't know when to reach for a tool, even when it's the obviously better path — a measurable, actionable failure mode
- **Fairness fix for a real problem**: With MCP adoption growing, comparing tool-enabled vs. GUI-only agents on GUI benchmarks is like comparing agents with and without internet access; OSWorld-MCP makes the playing field explicit
- **Multi-round tool chaining is a wall**: 4-round chaining tasks score 0% GUI-only and 16.7% with tools for the best model — compound failure rates reveal a capability cliff not visible in single-tool evaluations
- **Efficiency dimension via ACS**: First computer-use benchmark to quantify *how many steps* a task takes, not just whether it succeeds — tools reduce ACS significantly, capturing the VS Code extension example (4 GUI steps → 1 MCP call)
- **Distractor tools test selection quality**: 25 non-target tools in the tool list means the benchmark punishes agents that grab the wrong tool, testing precision not just recall of tool usage

## Limitations

- **Coverage**: 7 applications is still limited; enterprise/web-heavy workloads (browsers, Slack, email clients) are underrepresented
- **Tool quality ceiling**: 158 tools were partly auto-generated by o3 then manually validated — subtle bugs or edge cases in tool implementations could penalize agents unfairly or give false passes
- **Single-OS scope**: Built on Ubuntu/Linux via OSWorld; Windows/macOS behavior differences not captured
- **Contamination risk**: MCP tool descriptions in the benchmark may overlap with tools seen in training data, particularly for models like Claude 4 which was involved in MCP's design
- **Binary success metric**: Pass/fail doesn't capture partial progress — a task completed in 3 of 4 required tool steps gets same score as complete failure
- **Static tool set**: Real-world MCP ecosystems evolve rapidly; the 158 fixed tools may not reflect cutting-edge or domain-specific server availability
- **Small N for multi-round**: 153 multi-round tasks (~42%) is reasonable but per-round-count cell sizes get small quickly, making statistical claims fragile

## Related Concepts

[[LLM Benchmarks]] [[LLM evaluation]] — parent context for this work  
[[OSWorld]] — the base environment this benchmark extends  
[[WebArena]] — related dynamic GUI agent benchmark  
[[ToolLLM|Tool Use]] — the core capability being measured  
[[RLHF]] — training paradigm relevant to improving tool-use behavior  
[[Transformer]] — backbone of the multimodal models being evaluated
