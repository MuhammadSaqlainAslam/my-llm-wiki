---
created: "2026-06-03"
title: "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs"
authors: "Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun"
year: "2023"
arxiv: "2307.16789"
tags: [benchmarks, tool-use, evaluation, api, instruction-tuning, llm-agents]
tldr: "ToolBench covers 16,464 real-world REST APIs across 49 categories; fine-tuned ToolLLaMA reaches near-ChatGPT tool-use performance on the accompanying ToolEval benchmark."
citation_count: 1813
---

## TL;DR
ToolLLM is an end-to-end framework (data → training → evaluation) for teaching open-source LLMs to use real-world APIs. The centerpiece is **ToolBench**: 126,486 instruction-tuning instances spanning 16,464 REST APIs from RapidAPI, with 469,585 real API calls recorded. Fine-tuned on this data, ToolLLaMA (LLaMA backbone) matches ChatGPT on tool-use benchmarks and generalizes zero-shot to unseen APIs.

## The Problem
Existing instruction-tuning datasets for tool use had three compounding problems:
1. **Too few / fake APIs** — prior work used at most ~1,645 APIs, often synthetic ones with no real execution
2. **Single-tool only** — no multi-tool, multi-step scenarios that mirror real user tasks
3. **Weak planning** — ReACT-style linear reasoning fails on complex instructions; even GPT-4 has low pass rates without better search

The result: open-source LLMs (LLaMA, Alpaca, Vicuna) were drastically behind ChatGPT on anything requiring real API interaction. ToolLLM targets closing that gap without relying on closed-source model internals.

## What It Measures

### ToolBench Dataset
| Property | Value |
|----------|-------|
| APIs collected | 16,464 REST APIs |
| Tools (API groups) | 3,451 |
| RapidAPI categories | 49 |
| Total instances | 126,486 |
| Real API calls recorded | 469,585 |
| Avg. reasoning traces per instance | 4.0 |
| Scenario types | Single-tool, intra-category multi-tool, intra-collection multi-tool |

Construction pipeline:
- **API Collection**: Crawl RapidAPI Hub; capture name, description, parameters, code snippets, example responses
- **Instruction Generation**: Prompt ChatGPT (gpt-3.5-turbo-16k) to write diverse human instructions for sampled API subsets, across single- and multi-tool scenarios
- **Solution Path Annotation**: Use ChatGPT + novel **DFSDT** (Depth-First Search Decision Tree) algorithm to find valid API-call chains; DFSDT backtracks on failed paths unlike linear ReACT

### ToolEval (Automatic Evaluator)
Two metrics, both backed by ChatGPT judgment:
- **Pass Rate** ↑ — did the model successfully fulfill the instruction within a budget of API calls?
- **Win Rate** ↑ — pairwise quality comparison of two solution paths vs. ChatGPT-ReACT as baseline

ToolEval shows high correlation with human evaluation, enabling scalable, reproducible assessment.

### Neural API Retriever
Trained to select relevant APIs from the full 16,464-API pool given a natural language instruction — removes the unrealistic assumption that users pre-specify the right APIs.

## Key Results

### Pass Rate & Win Rate (vs. ChatGPT-ReACT baseline)
| Model | Strategy | Pass Rate | Win Rate |
|-------|----------|-----------|----------|
| Vicuna / Alpaca | ReACT | ~0.05–0.10 | ~0.10 |
| Text-Davinci-003 | ReACT | ~0.35 | ~0.35 |
| Text-Davinci-003 | DFSDT | ~0.45 | ~0.45 |
| Claude-2 | ReACT | ~0.40 | ~0.40 |
| Claude-2 | DFSDT | ~0.50 | ~0.50 |
| **ChatGPT** | ReACT | ~0.55 | 0.50 (baseline) |
| **ChatGPT** | DFSDT | ~0.65 | ~0.60 |
| **ToolLLaMA** | ReACT | ~0.45 | ~0.45 |
| **ToolLLaMA** | DFSDT | ~0.55 | ~0.55 |
| GPT-4 | ReACT | ~0.60 | ~0.65 |
| GPT-4 | DFSDT | ~0.70 | ~0.70 |

*Approximate values read from Figure 2; ToolLLaMA-DFSDT matches ChatGPT-ReACT and comes within ~5–10% of ChatGPT-DFSDT.*

### Out-of-Distribution: APIBench (Gorilla benchmark)
| Model | APIBench Performance |
|-------|----------------------|
| Gorilla (specialized) | Strong baseline |
| **ToolLLaMA** (zero-shot OOD) | On par with Gorilla |

ToolLLaMA was never trained on APIBench APIs or instructions — strong generalization signal.

### DFSDT vs. ReACT (Annotation Efficiency)
DFSDT enables completion of instructions that ReACT fails entirely on, by exploring multiple reasoning paths and backtracking. GPT-4 alone with ReACT achieves low pass rates on hard multi-tool instructions; with DFSDT annotation efficiency rises enough to make the dataset construction tractable.

## Why It Matters for LLM Evaluation

- **Scale gap exposed**: Prior tool-use benchmarks topped out at ~1,645 APIs (APIBench) or 53 (API-Bank); ToolBench's 16,464 APIs with 469,585 real execution traces reveals how brittle current models are at real-world API diversity
- **Multi-tool realism**: Only ToolBench covers intra-category and intra-collection multi-tool scenarios — the kind of task composition users actually want (e.g., "plan a movie night in the mountains" requiring weather + streaming + location APIs simultaneously)
- **DFSDT as reasoning baseline**: Demonstrates that the evaluation strategy (ReACT vs. tree search) dramatically changes measured model capability — models can look much weaker simply due to the planner, not their knowledge
- **Open-source parity proof**: Shows LLaMA-scale models can reach ~ChatGPT tool-use performance when trained on the right data, directly quantifying the training-data gap rather than architectural gap
- **Automatic evaluation validity**: ToolEval's ChatGPT-backed pass/win metrics correlate strongly with human judgments, providing a scalable evaluation protocol others can reuse without per-study human annotation

## Limitations

- **Data freshness / API churn**: RapidAPI APIs change, deprecate, or go behind paywalls — ToolBench snapshots may not reflect live API behavior, making longitudinal reproducibility hard
- **ChatGPT-generated labels**: Both instructions and solution paths are auto-generated by ChatGPT; systematic biases or errors in the "teacher" propagate into the benchmark
- **Evaluator circularity**: ToolEval uses ChatGPT as judge, and ToolLLaMA is trained to mimic ChatGPT — the model being evaluated and the evaluator share the same source, risking inflated scores
- **English / Western API bias**: RapidAPI skews heavily toward English-language, Western services; generalization to other ecosystems is untested
- **Closed execution environment**: Real API calls require live network access and valid API keys; benchmark reproduction requires RapidAPI subscriptions, limiting community replication
- **Saturation ceiling unclear**: GPT-4-DFSDT already achieves ~0.70 pass rate; as stronger models emerge, the headroom in the current test split may compress quickly

## Related Concepts
[[LLM Benchmarks]] [[LLM evaluation]] [[RLHF]] [[Transformer]]

**Related benchmarks**: APIBench (Gorilla), API-Bank, ToolAlpaca, HotpotQA (multi-step reasoning analog)
**Capabilities tested**: Tool use / API calling, multi-step planning, instruction following, retrieval-augmented generation, generalization to unseen APIs
**Methods**: ReACT prompting, Chain-of-Thought, depth-first search over reasoning trees, neural retrieval, supervised fine-tuning (SFT)
