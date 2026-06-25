---
created: "2026-06-24"
title: "MAI-Thinking-1: Building a Hill-Climbing Machine"
authors: "The Microsoft AI Team"
year: "2026"
arxiv: ""
technical_report: "https://microsoft.ai/pdf/mai-thinking-1.pdf"
source_type: "technical_report"
tags: [model-family, moe, reasoning, reinforcement-learning, microsoft, scaling]
tldr: "Microsoft AI's first from-scratch reasoning model — 35B active / 1T total parameter MoE trained on 30T tokens of exclusively licensed/public data with no distillation, achieving 52.8% SWE-Bench Pro and 97.0% AIME 2025, competitive with Sonnet 4.6"
citation_count: 0
---

## TL;DR

MAI-Thinking-1 is Microsoft AI's first reasoning model trained entirely from scratch — no distillation from any third-party model. A 35B active / 1T total parameter MoE, pretrained on 30T tokens of exclusively licensed and public data, then taken through a three-stage RL "climb" (STEM, agentic coding, helpfulness/safety) before being consolidated into a single model. Reaches 52.8% on SWE-Bench Pro, 97.0% on AIME 2025, and is described as competitive with Sonnet 4.6 across a wide benchmark range.

---

## Architecture

- Decoder-only Transformer with interleaved local/global attention (5:1 ratio, following Gemma 3's design) and alternating dense/MoE feed-forward blocks
- Adopts [[LatentMoE]] (compressing tokens before all-to-all expert dispatch), citing NVIDIA's design directly
- 8 of 512 experts activated per token
- GQA with 8 KV heads, RoPE on local attention layers, no position encoding on global attention layers
- o200k_base tokenizer (200,019 vocab)
- FP8 mixed precision (E4M3 forward, E5M2 data-gradient), fully deterministic training (bitwise-reproducible runs)
- Trained on 8,192 GB200 GPUs using YOLO, Microsoft's in-house training framework

---

## Training Recipe

**Pre-training:** 30T tokens, no synthetic or LM-generated data anywhere in the corpus. Explicit exclusion of huggingface.co and similar ML-repository domains to avoid benchmark contamination.

**Mid-training:** Two-stage context extension (64K then 256K tokens) with re-weighted, STEM/code-biased subsets of the same corpus.

**RL climb:** Starts from a checkpoint with zero prior exposure to reasoning traces. Uses a modified [[GRPO]] objective with adaptive entropy control and an outer ratio clip for stability. Three domain specialists are trained in parallel:

1. **STEM / competitive code** — math olympiad + competitive programming
2. **Agentic coding / tool use** — multi-step tool-calling, SWE-Bench-style tasks
3. **Helpfulness / safety** — instruction following and alignment

The three specialists are then consolidated into a single model via SFT, followed by a final lightweight RL stage to restore generality.

---

## Why It Matters

- A rare, unusually transparent technical report — documents a failure mode ("data-mixture rank non-invariance": a data mixture that wins at small scale can lose at large scale) that most vendor reports omit
- Explicit, deliberate choice to train without any distillation data, positioned as a design principle ("capabilities should be learned, not inherited") rather than a limitation
- Direct architectural lineage from papers already in this wiki: Gemma 3's local/global attention interleaving and NVIDIA's [[LatentMoE]]
- Reports BPB comparisons against [[DeepSeek-V3 Technical Report|DeepSeek-V3]], Kimi-K2, and Gemma 4 31B base models — useful cross-reference for benchmark comparisons elsewhere in the wiki
- First major reasoning model to publicly report SWE-Bench Pro (52.8%) alongside AIME 2025 (97.0%), offering a joint math+coding capability snapshot

---

## Limitations

- Self-reported; no independent evaluator benchmarks cited in the report
- "Competitive with Sonnet 4.6" is a vendor characterization — worth verifying against the full report's comparison table before citing specific numbers in other notes
- No arXiv submission — technical report only, same caveat as [[Nemotron_3_Super|Nemotron 3 Super]], [[Nemotron_3_Ultra|Nemotron 3 Ultra]], and [[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence|DeepSeek-V4]]

---

## Related Concepts

*Architecture: [[LatentMoE]] · [[Mixture-of-Experts]] · [[GRPO]]*

*Comparable models: [[Nemotron_3_Super|Nemotron 3 Super]] · [[Nemotron_3_Ultra|Nemotron 3 Ultra]] · [[DeepSeek-V3 Technical Report|DeepSeek-V3]]*

*Training: [[Multi-Environment RLVR Training]] · [[Chinchilla_Scaling_Laws|Chinchilla]]*
