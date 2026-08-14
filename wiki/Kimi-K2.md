---
created: "2026-07-06"
title: "Kimi K2: Open Agentic Intelligence"
authors: "Kimi Team, Moonshot AI"
year: "2025"
arxiv: "2507.20534"
tags: [model-family, moe, agentic, reasoning, moonshot]
tldr: "Moonshot AI's open MoE model specializing in agentic tasks — 1T total parameters, frontier performance on coding/math/tool-use, competitive with DeepSeek-V3 and Qwen3-235B on key benchmarks, released under a commercial-friendly license"
citation_count: 354
---

## TL;DR

Kimi K2 is Moonshot AI's fully open, commercially-licensed MoE model, purpose-built for agentic tasks: long-horizon tool use, multi-step reasoning, and complex coding. At 1T total parameters, it achieves 53.7% on LiveCodeBench v6 and competitive results on GPQA Diamond and AIME 2025, positioning itself alongside DeepSeek-V3 and Qwen3-235B as a top-tier open agentic model.

---

## Architecture

- Mixture-of-Experts (MoE) — 1T total parameters, active parameters undisclosed
- Purpose-trained for agentic task execution rather than pure language modeling
- Tool-calling and multi-step planning as primary design target

---

## Key Benchmark Results

| Benchmark | Kimi K2 | DeepSeek-V3 | Qwen3-235B | GPT-4.1 |
|-----------|---------|-------------|------------|---------|
| LiveCodeBench v6 | 53.7% | — | — | — |
| GPQA Diamond | 68.2% | 74.9% | 66.3% | 62.9% |
| AIME 2025 | 46.6% | 33.9% | 37.0% | 24.7% |

> Source: Kimi K2 technical report (arXiv:2507.20534). Non-thinking mode unless otherwise stated.

---

## Why It Matters

- Moonshot AI's "open-source gambit" — releasing a world-class model openly to compete with closed labs, following the same playbook as [[DeepSeek-V3 Technical Report|DeepSeek-V3]] and [[Qwen3 Technical Report|Qwen3]]
- Strong agentic benchmark results from an organization previously not among the top open-model providers
- Direct successor line (K2.5 in Jan 2026 with multimodal vision via MoonViT-3D, K2.6 in April 2026 with long-horizon agentic coding focus) confirms this is a sustained research direction, not a one-time release

---

## Limitations

- Technical details (exact expert count, per-token active parameters) not fully disclosed in the public report
- Benchmark comparisons use non-thinking mode throughout; thinking-mode comparisons would paint a different picture for math
- Independent verification of agentic benchmark results pending — these are first-party reported numbers

---

## Related Concepts

*Lineage: [[DeepSeek-V3 Technical Report|DeepSeek-V3]] · [[Qwen3 Technical Report|Qwen3]] · [[The Llama 3 Herd of Models|LLaMA 3]] · [[Mixtral]]*

Also from Moonshot AI's Kimi Team: **[[Attention Residuals]]**, integrated into the separate Kimi Linear architecture (48B/3B activated), not this model. Succeeded by **[[Kimi-K3|Kimi K3]]** (2.8T/104B activated), which builds on Attention Residuals directly.
