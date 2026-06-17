---
title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"
authors: "Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, Tao Yu"
year: "2024"
arxiv: "2404.07972"
tags: [benchmarks, computer-use, agentic, multimodal, evaluation]
tldr: "OSWorld is a real-computer-environment benchmark of 369 tasks across arbitrary OS apps where humans score 72.36% but the best model (GPT-4V) scores only 12.24%"
citation_count: 796
---

## TL;DR
OSWorld is the first scalable, truly interactive benchmark for multimodal computer-use agents, running tasks inside real virtual machines (Ubuntu, Windows, macOS) with execution-based evaluation scripts. It exposes a massive human-AI gap: humans complete 72.36% of tasks; the best evaluated model manages only 12.24%. This gap is primarily driven by failures in GUI grounding (clicking precise pixels) and lack of operational knowledge about real applications.

## The Problem
Previous computer-use benchmarks fail in one or more of three ways:
1. **No interactive environment** — they provide static demonstration datasets and evaluate via string matching, which wrongly penalizes valid alternative solutions and blocks interactive/RL-style learning.
2. **Application- or domain-locked** — web-only (WebArena, MiniWoB++), coding-only, or similar narrow scopes. Agents trained/evaluated there can't generalize to real multi-app desktop workflows.
3. **Simplified action/observation spaces** — prior executable environments abstract away raw mouse/keyboard into high-level APIs, hiding the actual difficulty of GUI grounding.

None of these benchmarks can test agents on the kind of open-ended, multi-application, cross-interface tasks real users perform daily.

## What It Measures

### Environment
- **Platform**: Real virtual machines (Ubuntu, Windows, macOS) controlled via raw keyboard/mouse API and CLI
- **Observation space**: Screenshots, accessibility (a11y) trees, or both
- **Action space**: Raw mouse clicks/drags, keyboard hotkeys, terminal commands, plus `DONE`/`FAIL` signals
- **Task formalism**: Partially Observable MDP (POMDP); max 15 steps per episode
- **Evaluation**: Custom execution-based scripts check final VM state (file contents, UI state, etc.) — binary 0/1 or partial credit

### Benchmark
- **369 tasks** derived from real user workflows, annotated by 9 CS-background authors
- **134 unique evaluation functions** — orders of magnitude more than prior work
- **Application categories**:
  - Web browsers (Chrome)
  - Desktop productivity (LibreOffice Calc, Writer, Impress; VS Code; GIMP; VLC; Thunderbird)
  - OS file I/O
  - Multi-app workflows spanning 2+ applications
- **Task types**: Single-app tasks + multi-app workflow tasks
- **Scoring**: Success Rate (Pass@1 equivalent) — fraction of tasks where execution-based eval returns reward = 1

## Key Results

| Model | Overall Success Rate | Notes |
|-------|---------------------|-------|
| Human | 72.36% | Baseline ceiling; tasks are time-consuming |
| GPT-4V (best config) | 12.24% | Best model overall |
| Claude-3 Opus | ~10% | Competitive with GPT-4V |
| Gemini series | <10% | Both Pro and Ultra variants |
| Qwen-Max | ~5% | Proprietary Chinese model |
| Mixtral / Llama-3 | <5% | Open-source LLMs |
| CogAgent | ~0.99% | Open-source VLM, near floor |
| Any model on multi-app workflows | ≤6.57% | Hardest subset |
| Any model on some app subsets | 0% | Complete failure on certain apps |

**Key ablations:**
- Adding accessibility (a11y) tree helps some models but actively misleads others — effect is model-dependent
- Set-of-Mark prompting provides mixed results
- Higher resolution input roughly **doubles** success rate on screenshot-based tasks but demands longer context

## Why It Matters for LLM Evaluation

- **Real execution, not string matching**: 134 custom eval scripts check actual VM state (file diffs, UI state, cookies, etc.), eliminating false negatives from valid alternative solutions — a fundamental methodological improvement over dataset-only benchmarks
- **The 60-point human-AI gap is a genuine signal**: 72.36% vs. 12.24% on tasks normal users can accomplish, showing frontier VLMs are far from practical computer assistants despite strong performance on static benchmarks
- **Multi-app workflows expose compositional failure**: best model drops to 6.57% on tasks requiring coordination across two or more applications, revealing that sequential single-app capability doesn't compose
- **GUI grounding is the key bottleneck**: analysis shows models systematically fail to predict precise pixel coordinates for clicks, tend toward repetitive action loops, and are confused by unexpected popup windows — failure modes invisible in text-only or simplified-action benchmarks
- **Cross-OS and cross-app scope enables scalability**: built on real VMs with config-file-driven setup, any researcher can add new tasks without building a custom simulator, making it a living benchmark rather than a fixed dataset

## Limitations

- **Contamination risk is low but not zero**: tasks are based on real use cases but the specific initial states and exact evaluation scripts are novel; however, applications themselves (LibreOffice, Chrome) appear in training data
- **369 tasks is relatively small**: covers many apps but individual app subsets can be <20 tasks, making per-app results noisy
- **Max 15 steps**: real computer tasks often require far more interactions; this cap may artificially limit agent strategies and penalize legitimate multi-step approaches
- **VM overhead**: running real VMs makes evaluation slow and computationally expensive compared to text benchmarks, limiting large-scale hyperparameter sweeps
- **English-centric and US-locale**: tasks and apps default to English interfaces; cross-lingual or locale-specific behavior untested
- **Partial credit is rare**: most tasks are binary success/fail, which may undercount meaningful partial progress and makes signal sparse for RL-based training
- **Evaluation script brittleness**: 134 custom scripts are carefully written but edge cases in task completion may still be misclassified; gold states are generated by authors, not crowdsourced at scale

## Related Concepts
[[LLM Benchmarks]] [[LLM evaluation]] [[Transformer]] [[RLHF]] [[RLVR]]

**Related benchmarks**: WebArena, MiniWoB++, Mind2Web, AgentBench, SWE-bench, ScreenSpot
**Capabilities tested**: GUI grounding, visual question answering, multi-step planning, tool use, cross-application reasoning
**Agent architectures**: ReAct-style agents, VLM-based agents, accessibility-tree augmented agents
