---
title: Qwen3 Technical Report
authors: Qwen Team
year: 2025
arxiv: 2505.09388
tags: [foundational, moe, inference, attention, benchmarks]
citation_count: 5861
tldr: Qwen3 unifies thinking and non-thinking reasoning modes into a single model family (0.6B–235B parameters) trained on 36 trillion tokens, with the flagship 235B-A22B MoE model scoring 85.7 on AIME'24 and 70.7 on LiveCodeBench v5.
---

## The Problem

The modern LLM ecosystem has fragmented into two distinct camps: fast chat models (like GPT-4o) optimized for snappy, context-driven responses, and slow reasoning models (like o1, QwQ-32B) that think long and hard via chain-of-thought. If you want both, you need to maintain two separate models, switch between them based on task complexity, and eat the engineering overhead of managing two different deployment pipelines. This is wasteful and inflexible.

Beyond the thinking/non-thinking split, there's a second practical problem: as you scale up flagship models, the computational cost of training smaller companions from scratch becomes disproportionately expensive relative to their performance gains. Doing full RL training on a 1.7B model just to get decent reasoning performance is inefficient when you already have a powerful 235B teacher sitting there.

Finally, existing open-weight models lagged in multilingual coverage. Qwen2.5 supported 29 languages — respectable, but inadequate for truly global deployment. Real-world applications increasingly demand coverage across dozens of language families simultaneously.

## The Idea

The core insight is that thinking and non-thinking are not fundamentally different model behaviors — they're two operating modes of the same underlying capability, and a single model can learn to switch between them on demand based on a simple prompt/template signal.

Unpacked: during post-training, Qwen3 is exposed to data both with and without reasoning chains in the same fine-tuning stage, then reinforcement learning is applied across both modes. The model learns that a "think" tag in the prompt means "spend tokens on scratchpad reasoning before answering," while its absence means "answer directly." A thinking budget mechanism then lets users cap how many tokens the model can spend reasoning — giving a continuous dial between latency and quality rather than a hard binary switch.

## How It Works

**Architecture.** Dense models (0.6B to 32B) follow the Qwen2.5 blueprint: Grouped Query Attention (GQA), SwiGLU activations, Rotary Positional Embeddings (RoPE), and RMSNorm with pre-normalization. Two changes from Qwen2: QKV-bias is dropped, and QK-Norm is added to stabilize training at scale. MoE models (30B-A3B and 235B-A22B) layer on fine-grained expert segmentation with 128 total experts and 8 activated per token. Notably, shared experts are removed compared to Qwen2.5-MoE, and a global-batch load balancing loss encourages genuine expert specialization rather than expert collapse.

**Pre-training (3 stages).** Stage 1: ~30 trillion tokens of broad general knowledge. Stage 2: knowledge-intensive data emphasizing STEM and code to sharpen reasoning. Stage 3: long-context data to push the context window from 4K to 32K tokens (128K for larger models). Data is expanded by running Qwen2.5-VL over massive PDF corpora for OCR, then cleaning with Qwen2.5 — yielding trillions of additional high-quality tokens. Synthetic math and code data come from Qwen2.5-Math and Qwen2.5-Coder respectively. Total: 36 trillion tokens across 119 languages.

**Post-training (4 stages).** Stage 1: Long chain-of-thought cold-start supervised fine-tuning to teach the model how to reason at all. Stage 2: RL on math and coding tasks to sharpen reasoning quality. Stage 3: Unified SFT mixing data with and without reasoning paths — this is what teaches the model to do both modes. Stage 4: General-domain RL to polish performance across the full task distribution. For smaller models, strong-to-weak distillation (both off-policy and on-policy) from larger Qwen3 models replaces RL as the primary capability transfer mechanism — it's faster and outperforms RL for small models.

**Thinking budget.** At inference time, users can set a maximum token budget for the internal scratchpad. Empirically, increasing the budget monotonically improves performance on hard tasks — you get an inference-time scaling curve essentially for free.

## Key Results

- **AIME'24**: 85.7 (flagship Qwen3-235B-A22B), competitive with o3-mini and DeepSeek-R1
- **AIME'25**: 81.5
- **LiveCodeBench v5**: 70.7
- **CodeForces rating**: 2,056
- **BFCL v3 (tool/agent tasks)**: 70.8
- Multilingual support: expanded from 29 languages (Qwen2.5) to **119 languages and dialects**
- Thinking budget experiments show consistent monotonic performance improvement as more thinking tokens are allocated
- Smaller models in the series achieve "highly competitive" performance vs. same-scale open-weight models, attributed to distillation from the flagship

## Limitations

- The unified thinking/non-thinking framework still requires users to correctly set the mode via prompt templates — there's no fully automatic task-complexity detection built in yet
- The 235B-A22B flagship requires substantial hardware even for MoE inference (22B activated params per token is still large)
- Thinking budget control is coarse — users set a token cap, but the model doesn't necessarily use the budget optimally for every subtask within a query
- Long-context performance (128K window) is enabled in larger models but the smallest models (0.6B, 1.7B) are capped at 32K
- Distillation-heavy training of smaller models means their reasoning style is inherited from the teacher — they may not generalize as well to reasoning patterns outside the teacher's distribution
- No multimodal capability in Qwen3 itself (vision understanding remains in the separate Qwen2.5-VL line)

## Why It Matters

Qwen3 matters for two reasons. First, it democratizes the reasoning vs. speed tradeoff: developers no longer need to pick between a fast chat model and a slow reasoning model — they get both in one open-weight package under Apache 2.0. This dramatically simplifies production pipelines for applications that need both capabilities (e.g., a coding assistant that answers simple questions instantly but thinks hard about algorithmic design).

Second, the thinking budget mechanism is an early concrete implementation of inference-time compute scaling that's user-controllable and hardware-agnostic. Unlike approaches that require special decoding infrastructure, this is just token budgeting — any standard inference stack can support it. Combined with the strong distillation pipeline that makes small models punch well above their weight, Qwen3 establishes a practical template for how future model families might be built: train one giant model extremely well, then cascade its knowledge downward through the size hierarchy.

## See Also

[[Qwen2.5]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[Mixture of Experts]] · [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain-of-Thought Prompting]] · [[GQA|Grouped Query Attention]] · [[RoPE|Rotary Position Embedding]] · [[RLHF|Reinforcement Learning from Human Feedback]] · [[Inference-Time Scaling]]
