---
created: "2026-06-10"
title: The Llama 3 Herd of Models
authors: Llama Team, AI @ Meta
year: 2024
arxiv: 2407.21783
tags: [foundational, inference, attention, benchmarks]
citation_count: 18076
tldr: Meta releases Llama 3, a herd of dense Transformer language models at 8B, 70B, and 405B parameters trained on 15T tokens that matches GPT-4 quality on key benchmarks while being fully open-source.
---

## The Problem

Previous open-weight models consistently lagged behind closed frontier models like GPT-4 by a meaningful margin, leaving the research community without a capable, publicly available baseline to build on. Llama 2, Meta's prior best effort, was trained on only 1.8T tokens — far below what scaling laws suggest is optimal for its parameter count — and its largest model topped out at 70B parameters with an 8K context window, limiting its usefulness on long-document tasks.

The field also lacked a single open model that could natively handle the combination of multilinguality, long-context reasoning (128K tokens), tool use, and code generation all at once. Practitioners had to stitch together specialized models or accept closed APIs, neither of which is ideal for reproducibility, safety research, or downstream innovation.

Finally, scaling to a 405B parameter model introduces enormous engineering complexity: training instability, data quality at 15T token scale, efficient inference serving, and post-training alignment without resorting to fragile RL algorithms all had to be solved simultaneously.

## The Idea

The core insight is that three levers — **data quality, scale, and managed complexity** — compound multiplicatively, and that aggressively optimizing all three at once, while deliberately choosing simpler architectural and training choices, produces a frontier-quality open model. Don't be clever with the architecture; be disciplined about the data and the compute.

Rather than reaching for a Mixture-of-Experts architecture or complex RL-based alignment, Meta doubled down on a standard dense Transformer, extremely clean pretraining data, and a straightforward SFT → Rejection Sampling → DPO post-training pipeline. The bet is that boring-but-scalable beats clever-but-fragile at this scale.

## How It Works

**Architecture** — Llama 3 is a dense Transformer (Vaswani et al., 2017) with minor adaptations. No mixture-of-experts routing, no exotic attention variants. The 405B flagship model has a 128K token context window achieved by first pretraining at 8K context and then extending via continued pretraining.

**Pretraining** — The model is trained on ~15.6T multilingual text tokens, roughly 8× more tokens than Llama 2. Total compute for the 405B model is 3.8 × 10²⁵ FLOPs — about 50× more than Llama 2's largest model. Smaller models (8B, 70B) are deliberately trained *longer* than compute-optimal, so they punch above their weight at inference time. The flagship model then acts as a teacher to further improve smaller models during post-training.

**Post-training pipeline** — Three stages:
1. **Supervised Fine-Tuning (SFT)** on high-quality instruction-following data.
2. **Rejection Sampling (RS)** — sample multiple completions, keep the best ones, re-train.
3. **Direct Preference Optimization (DPO)** — align to human preferences without the instability of PPO-style RL.

This pipeline is intentionally simple. Complex RL algorithms (PPO, RLHF with a separate reward model) are avoided because they are harder to scale and less stable.

**Multimodal extensions (not yet released)** — Image and speech encoders are trained separately and fused into the language model via cross-attention adapter layers. The image encoder is trained on image-text pairs; the speech encoder uses self-supervised masked reconstruction over discrete token representations.

**Safety** — Llama Guard 3 is released alongside the base models to perform input/output safety classification.

## Key Results

On MMLU (5-shot), the 405B model scores **87.3**, compared to GPT-4 (0125) at 85.1 and GPT-4o at 89.1 — essentially on par with the best closed models. Selected highlights:

| Benchmark | Llama 3 8B | Llama 3 70B | Llama 3 405B | GPT-4o |
|---|---|---|---|---|
| MMLU (5-shot) | 69.4 | 83.6 | 87.3 | 89.1 |
| HumanEval (0-shot) | 72.6 | 80.5 | 89.0 | 90.2 |
| GSM8K (8-shot, CoT) | 84.5 | 95.1 | 96.8 | 96.1 |
| MATH (0-shot, CoT) | 51.9 | 68.0 | 73.8 | 76.6 |
| GPQA (0-shot, CoT) | 32.8 | 46.7 | 51.1 | 53.6 |
| MGSM multilingual | 68.9 | 86.9 | 91.6 | 90.5 |

The 8B and 70B models are described as best-in-class in their parameter brackets, outperforming Mistral 7B, Gemma 2 9B, and Mixtral 8x22B respectively. The 405B model is competitive with Claude 3.5 Sonnet and GPT-4o across most categories.

## Limitations

- **Multimodal models not released**: Image, video, and speech-capable versions are still under active development as of July 2024 and are not publicly available.
- **Context at pretraining vs. fine-tuning**: Initial pretraining uses only 8K context; 128K is achieved through continued pretraining, which may mean the model is less naturally adapted to very long contexts compared to architectures designed for it from scratch.
- **Dense architecture at 405B is expensive to serve**: Choosing a dense Transformer over MoE means inference costs scale linearly with parameters. Running 405B at low latency requires significant hardware investment, limiting practical deployment.
- **Compute-optimal tradeoff**: Smaller models are trained beyond compute-optimal to maximize inference-time quality, but this means training costs are higher than necessary if you only care about the smaller models.
- **Still behind on hard reasoning**: GPQA scores (measuring PhD-level science questions) top out at 51.1 for the 405B, while Claude 3.5 Sonnet reaches 59.4, suggesting the frontier of hard reasoning is not yet matched.

## Why It Matters

Llama 3 is arguably the most important open-weight model release to date. A 405B parameter model matching GPT-4 quality, released with open weights and a permissive community license, fundamentally changes the economics of AI research. It means:

- **Academic labs** can now reproduce near-frontier experiments without paying API costs.
- **Fine-tuning and alignment research** has a strong public baseline to improve upon.
- **Safety research** benefits from having the actual weights available for red-teaming and interpretability work.
- The release also validates the scaling-laws-plus-simple-design philosophy: you don't need architectural exotica to get frontier performance, just disciplined data curation and enough compute.

The decision to release Llama Guard 3 alongside the base models sets a template for bundled safety tooling with open releases.

## See Also

[[Transformer|Attention Is All You Need]] · [[Llama 2]] · [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|Direct Preference Optimization]] · [[Scaling Laws for Neural Language Models]] · [[Mixture-of-Experts|Mixtral of Experts]] · [[GPT-4 Technical Report]] · [[Transformer]]
