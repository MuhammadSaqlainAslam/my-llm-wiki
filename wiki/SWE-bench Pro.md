---
created: "2026-06-03"
title: "SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?"
authors: "Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, et al."
year: "2025"
arxiv: "2509.16941"
tags: [benchmarks, coding, agentic, evaluation, software-engineering, contamination-resistant]
tldr: "SWE-Bench Pro is a 1,865-problem enterprise-grade software engineering benchmark across 41 repositories where top models score below 45% Pass@1, designed to resist contamination and capture multi-file, long-horizon complexity beyond SWE-Bench Verified."
citation_count: 130
---

## TL;DR
SWE-Bench Pro is Scale AI's harder successor to SWE-Bench, featuring 1,865 problems from 41 actively maintained repositories (including proprietary startup codebases) that require multi-file patches averaging 107.4 lines across 4.1 files. It is explicitly designed to resist data contamination via GPL licensing and commercial codebase acquisition. Top models max out below 45% Pass@1, leaving substantial headroom for progress.

## The Problem
SWE-Bench Verified is saturating and was never representative of real enterprise software engineering:

- **Contamination**: Public GitHub repos under permissive licenses (MIT, Apache, BSD) are routinely scraped into LLM pretraining corpora, making benchmark integrity questionable.
- **Triviality**: 161 of 500 SWE-Bench Verified problems require only 1–2 line changes — not representative of industrial tasks that demand multi-file, multi-hundred-line modifications.
- **Narrow scope**: SWE-Bench draws from 12 Python repos; real enterprise codebases are polyglot, proprietary, and domain-diverse.
- **No long-horizon testing**: Existing benchmarks don't capture tasks that would take a professional engineer hours to days to complete.

## What It Measures

**Scale**: 1,865 total problems across 41 repositories, partitioned into:
- **Public set** (11 repos, open access): GPL-licensed, fully released
- **Held-out set** (12 repos, private): Reserved for overfitting detection
- **Commercial set** (18 repos, private): Purchased from real startups; results released but code stays private

**Complexity requirements**:
- Every problem requires ≥10 lines of code change (trivial 1–10 line edits excluded)
- Average solution: **107.4 lines of code across 4.1 files**
- 100+ tasks require >100 lines of modification
- Tasks span multiple files by design

**Task structure**: Each instance contains:
1. Human-rewritten **problem statement** (from original commits/PRs/issues + added context)
2. **Requirements** list — explicit, grounded in unit tests, resolves ambiguity
3. **Interface specification** — class/function names expected by tests to eliminate false negatives from valid-but-wrong-interface solutions

**Languages**: Python (isolated venvs), JavaScript/TypeScript (Node.js + npm/yarn), Go (module-aware environments) — all containerized as Docker images for reproducibility

**Scoring**: Pass@1 — patch must resolve the issue and pass a human-reviewed fail2pass test suite

**Contamination resistance**:
- Public/held-out sets: only GPL/copyleft repos (legal barriers to inclusion in commercial training data)
- Commercial set: private codebases purchased directly from startups

**Human verification**: Three-stage human-in-the-loop pipeline to (1) clarify ambiguity and add missing context, and (2) recover/write unit tests that constrain solution space without false negatives

**Repository balance**: Each repo contributes 50–100 instances (hard cap at 100) to prevent models from gaming the benchmark by overfitting to a single codebase's idioms

## Key Results

| Model | Score (Pass@1) | Notes |
|-------|---------------|-------|
| Best evaluated model | <45% | Under unified scaffold (SWE-agent) |
| SWE-Agent baseline | ~reported in paper | Public subset, standardized scaffold |
| All evaluated coding models | <45% | Consistent ceiling across models |

*(Full per-model breakdown is available at scale.com/research/swe_bench_pro; paper emphasizes the <45% ceiling as the headline finding)*

**Contrast with SWE-Bench Verified**: Leading models now solve 50–70%+ of SWE-Bench Verified, making it effectively saturated for frontier model comparison. SWE-Bench Pro preserves meaningful headroom.

## Why It Matters for LLM Evaluation

- **Contamination-resistant by construction**: GPL licensing creates legal friction against inclusion in commercial training corpora; commercial startup codebases are by definition not in public web crawls — two independent contamination defense layers.
- **Realistic complexity floor**: Minimum 10-line changes, average 107.4 lines across 4.1 files — much closer to what enterprise engineers actually do than SWE-Bench's distribution of 1–2 line fixes.
- **Long-horizon capability probe**: Tasks are scoped to require hours–days of professional engineer effort, directly testing the autonomous, multi-step reasoning that agentic systems must master.
- **Failure mode taxonomy**: The authors cluster error patterns from collected agent trajectories, providing a diagnostic lens on *where* and *why* current models fail — not just that they fail.
- **Three-tier access policy**: Public (research), held-out (overfitting detection), commercial (private codebases) — a structural solution to the benchmark lifecycle problem of eventual saturation and gaming.
- **Polyglot and domain-diverse**: Python, JS/TS, and Go across consumer apps, B2B platforms, and developer tools — avoiding the Python-only monoculture of original SWE-Bench.

## Limitations

- **<45% ceiling currently, but unknown floor**: If agents improve rapidly, the held-out and commercial sets provide a buffer, but the public set could saturate.
- **Human augmentation introduces subjectivity**: Requirements and interface specs are human-written; different annotators might produce different ground truths for the same underlying issue.
- **Test suite quality depends on human reviewers**: False negatives from poorly written tests remain a risk even with the three-stage verification process; the interface spec mitigation is partial.
- **Commercial set opacity**: Results on the commercial set are released without the code, making it impossible for external researchers to diagnose failures or verify evaluation correctness.
- **GPL-as-contamination-proxy is imperfect**: GPL repos do appear in some training sets (e.g., The Stack dataset includes GPL code); legal barrier ≠ guaranteed absence from training data.
- **Unified scaffold bias**: All evaluations use a single scaffold (SWE-agent); model rankings could shift with scaffold optimization, potentially conflating scaffold quality with model capability.
- **Repository cap of 100 helps but doesn't eliminate repo-specific advantages**: Models pretrained on specific startup tools could still have unfair advantages on commercial set problems.
- **No cross-repository dependency tasks**: Real enterprise engineering often involves issues spanning multiple codebases/services; SWE-Bench Pro still operates within single-repo boundaries.

## Related Concepts

[[LLM Benchmarks]] · [[LLM evaluation]] · [[RLVR]] · [[RLHF]]

**Directly related benchmarks**:
- SWE-Bench / SWE-Bench Verified (predecessor)
- SWE-Bench Lite
- Multi-SWE-bench (polyglot extension)
- HumanEval (function-level code generation)
- MBPP (crowd-sourced Python problems)
- APPS (10K algorithmic programming problems)

**Agent frameworks evaluated**:
- SWE-agent (primary scaffold used)
- AutoCodeRover (AST-based code search agent)

**Capabilities tested**:
- Long-horizon agentic planning
- Multi-file code editing and refactoring
- Codebase-scale understanding
- Test-driven development
- Repository navigation and search
