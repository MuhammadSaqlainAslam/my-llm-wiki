---
created: "2026-06-03"
title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
authors: "Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan"
year: "2024"
arxiv: "2310.06770"
tags: [benchmarks, coding, agentic, evaluation, software-engineering]
tldr: "A benchmark of 2,294 real GitHub issues requiring repository-scale code edits, where the best model at launch (Claude 2) solved only 1.96%"
citation_count: 3528
---

## TL;DR
SWE-bench tasks LMs with resolving real GitHub issues by editing large Python codebases — not generating standalone snippets, but navigating thousands of files, understanding cross-function dependencies, and producing patches that pass actual CI tests. At launch, even Claude 2 solved only 1.96% of problems, making it one of the hardest practical coding benchmarks in existence.

## The Problem
Existing coding benchmarks (HumanEval, MBPP) test self-contained, few-line problems that bear little resemblance to real software engineering. Meanwhile, frontier models were saturating these benchmarks, making them useless for tracking progress. Real bugs require: navigating codebases with 3,000+ files, coordinating changes across multiple functions and files, and producing solutions that survive existing regression test suites — none of which prior benchmarks tested.

## What It Measures

**Scale:**
- **2,294 task instances** sourced from 12 popular Python repos (Django, scikit-learn, matplotlib, sympy, pytest, etc.)
- **300-instance Lite subset** for faster iteration, focused on self-contained functional bugs
- **19,000 training instances** (SWE-bench-train) from 37 repos for fine-tuning experiments

**Task structure:**
- Input: issue text (avg. 195 words, max 4,477) + full codebase snapshot (avg. 438K lines, 3,010 files)
- Output: a patch file specifying line-level edits
- Reference solutions edit avg. **1.7 files, 3.0 functions, 32.8 lines**

**Evaluation:**
- Apply patch via unix `patch`, run repository's real test suite
- Must pass all **fail-to-pass tests** (at least 1 per instance, avg. 9.1) AND not break existing passing tests (median 51 regression tests)
- Metric: **% of issues fully resolved** (binary pass/fail per instance)

**Construction pipeline:**
1. Scrape ~90,000 PRs from 12 repos
2. Filter to PRs that (a) reference a GitHub issue and (b) modify test files
3. Execution-filter: keep only instances with ≥1 test going fail→pass without errors
4. 90,000 → **2,294 instances** survive all stages

## Key Results

| Model | % Resolved (Full) | Retrieval Method | Notes |
|-------|-------------------|------------------|-------|
| Claude 2 | **1.96%** | BM25 | Best at launch |
| SWE-Llama 13b | ~1.5% | Oracle (gold files) | Competitive w/ Claude 2 in oracle setting |
| SWE-Llama 7b | <1% | BM25 | Fine-tuned CodeLlama |
| GPT-4 | <2% | BM25 | Similar to Claude 2 |
| ChatGPT (GPT-3.5) | <1% | BM25 | Near zero |

> **Note:** Oracle retrieval (feeding model exactly the right files to edit) substantially boosts performance, revealing that retrieval is a major bottleneck separate from code editing capability. Post-publication leaderboard entries (e.g., SWE-agent, Devin) have pushed resolved rates to ~12-50%+ as agentic frameworks matured — the benchmark is doing its job.

## Why It Matters for LLM Evaluation

- **Near-zero baseline scores** prevent saturation for years: best models at launch solved <2% of 2,294 real issues, providing enormous headroom to measure genuine progress
- **Execution-based, unfakeable evaluation**: solutions must pass real test suites — no LLM judge, no multiple choice, no ambiguity; a patch either fixes the bug or it doesn't
- **Contamination-resistant by design**: the benchmark can be continuously extended with issues filed *after* model training cutoffs, making it hard to memorize
- **Tests full-stack software engineering**, not just code generation: models must do file navigation, multi-location edits, understanding of existing APIs and test frameworks — skills invisible to HumanEval-style benchmarks
- **Enables agentic evaluation**: the open-ended patch output format accommodates retrieval pipelines, long-context models, and tool-using agents on equal footing — a natural testbed for autonomous coding agents

## Limitations

- **Retrieval is a major confounder**: performance depends heavily on whether the model is given the right files (oracle) vs. having to find them (BM25). This means scores reflect retrieval quality as much as code editing ability
- **Python-only**: all 12 repos are Python packages; generalization to other languages (C++, Java, Rust) is untested and the construction pipeline would need adaptation
- **Repository popularity bias**: focuses on well-maintained, popular PyPI packages with good test coverage — may not reflect messier real-world codebases
- **Binary resolution metric**: a patch that fixes 9/10 fail-to-pass tests scores the same as one that fixes 0/10 — no partial credit, which can make optimization noisy
- **Compute cost**: full evaluation requires spinning up Docker containers per instance and running full test suites, making it expensive to iterate on (motivating the 300-instance Lite split)
- **Test suite as ground truth**: if the reference tests are incomplete or wrong, a valid alternative solution could be marked incorrect

## Related Concepts
[[LLM Benchmarks]] · [[LLM evaluation]] · [[RLHF]] · [[RLVR]] · HumanEval · CodeLlama · BM25 Retrieval · Agentic LLMs · Execution-based Evaluation · SWE-agent · Devin
