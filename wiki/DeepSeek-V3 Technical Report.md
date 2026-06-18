---
title: DeepSeek-V3 Technical Report
aliases: ["DeepSeek-V3"]
authors: DeepSeek-AI
year: 2024
arxiv: 2412.19437
tags: [moe, attention, inference, hardware, foundational, training]
citation_count: 3500
tldr: DeepSeek-V3 is a 671B MoE model (37B active per token) that matches GPT-4o and Claude-3.5-Sonnet on major benchmarks while costing only $5.576M (~2.788M H800 GPU hours) to train fully.
---

## The Problem

Training frontier-scale language models is extraordinarily expensive. A model like GPT-4 reportedly cost tens or hundreds of millions of dollars to train, making it inaccessible to most research labs and companies. The open-source ecosystem had strong models — LLaMA-3.1-405B, Qwen2.5-72B — but none that could seriously challenge the best closed-source models on hard benchmarks like MATH 500 or competitive coding.

The fundamental tension is: more parameters = better performance, but also more compute per forward pass, more memory, more communication overhead across GPUs, and more training instability. Dense models pay full compute cost for every parameter on every token, which is wasteful. Sparse MoE models solve the compute problem but introduce nasty load-balancing issues — if some experts get hammered and others sit idle, you waste hardware and hurt gradient signal. Previous load-balancing fixes added auxiliary losses to the training objective, which helped hardware utilization but hurt model quality.

At inference time, large MoE models across many GPUs also suffer from crushing all-to-all communication overhead when routing tokens to experts on different nodes. Prior work largely accepted this as unavoidable, or limited expert routing within a single node.

## The Idea

The core insight is that algorithmic co-design — architecture, training framework, and hardware utilization all optimized together — can let you train a 671B-parameter model for roughly the cost of a nice house, without sacrificing quality. Specifically: use MLA to compress the KV cache, use DeepSeekMoE with fine-grained expert routing for efficient sparse compute, drop the auxiliary loss for load balancing in favor of a bias-term trick, add multi-token prediction, and build a custom pipeline parallelism algorithm (DualPipe) that hides communication behind computation almost entirely.

## How It Works

**Architecture: Multi-head Latent Attention (MLA)**
Standard multi-head attention stores a full KV cache: for every layer and every token in context, you keep K and V vectors. At long contexts this dominates memory. MLA instead compresses K and V into a low-dimensional latent vector and reconstructs them at inference time. This dramatically shrinks the KV cache footprint, enabling efficient long-context inference without the memory explosion.

**Architecture: DeepSeekMoE**
Instead of a dense FFN, each transformer layer routes tokens to a small subset of many fine-grained experts. DeepSeek-V3 has 671B total parameters but only 37B activate per token — roughly a 5.5× savings in active compute versus a dense model of equivalent parameter count. The experts are fine-grained (more, smaller experts rather than fewer, larger ones), which gives better specialization and flexibility.

**Auxiliary-Loss-Free Load Balancing**
The traditional trick to keep experts balanced is to add a penalty term to the loss that punishes uneven routing. This works but fights against the model — it's trying to learn good routing while also being punished for following that routing if it concentrates. DeepSeek-V3 instead adds a learnable per-expert bias term to the routing scores. If an expert is getting over-used, its bias gets nudged down; if under-used, nudged up. This keeps load balanced without ever touching the gradient of the main loss. Result: cleaner gradients, better final model quality.

**Multi-Token Prediction (MTP)**
Instead of predicting only the next token, the model is trained to predict the next *D* tokens at each position simultaneously using additional output heads. The main causal chain remains intact — MTP heads are sequential, each consuming previous predictions — but training signal is richer. At inference, MTP heads can also serve as a speculative decoding mechanism, generating candidate tokens that the main model then verifies, boosting throughput.

**FP8 Mixed Precision Training**
Most large model training uses BF16. DeepSeek-V3 introduces FP8 for compute and storage — the first validation of this at truly extreme scale (671B params). FP8 has a tiny dynamic range, so naive application fails. Their solution: fine-grained block-wise quantization with separate scaling factors per tile, and careful handling of which operations stay in higher precision (e.g., embedding tables, loss computation). Net result: faster matmuls, lower memory bandwidth pressure.

**DualPipe: Hiding Communication Behind Compute**
In pipeline parallelism, the classic problem is the "bubble" — GPUs sit idle waiting for activations to arrive from the previous stage. DualPipe overlaps forward and backward passes from different micro-batches so computation is nearly always happening while communication occurs in the background. For MoE across nodes, all-to-all expert routing (sending tokens to the right expert on another machine) is the bottleneck. DualPipe + custom InfiniBand/NVLink kernels reduce this to near-zero effective overhead, as long as the computation-to-communication ratio is maintained.

**Training Recipe**
- Pre-train on 14.8T diverse, high-quality tokens
- Two-stage context extension: first to 32K, then to 128K
- Post-train with SFT then RL (Group Relative Policy Optimization, GRPO)
- Distill reasoning capabilities from DeepSeek-R1 during post-training

## Key Results

**Cost:** 2.788M H800 GPU hours total (~$5.576M at $2/hr). Each trillion pre-training tokens costs 180K GPU hours — about 3.7 days on 2048 H800s. Full pre-training completed in under two months.

**Stability:** Zero irrecoverable loss spikes, zero rollbacks across the entire 14.8T token pre-training run. This is remarkable at this scale.

**Benchmarks (chat model):**
- MATH 500: **90.2%** (vs Claude-3.5-Sonnet 78.3%, GPT-4o-0513 74.6%)
- AIME 2024: **39.2%** (vs Claude-3.5-Sonnet 16.0%, GPT-4o 9.3%)
- GPQA Diamond: **59.1%** (vs Claude-3.5-Sonnet 65.0%, GPT-4o 49.9%)
- Codeforces Percentile: **51.6%** (vs Claude-3.5-Sonnet 20.3%)
- SWE-bench Verified: **42.0%** (vs Claude-3.5-Sonnet 50.8%)
- MMLU-Pro: **75.9%** (vs Claude-3.5-Sonnet 78.0%)

DeepSeek-V3 is the strongest open-source model across nearly all benchmarks tested, and trades blows with or beats GPT-4o and Claude-3.5-Sonnet on most tasks — particularly math and coding.

## Limitations

- **Hardware dependency:** The infrastructure optimizations (DualPipe, custom all-to-all kernels) are tightly coupled to their specific cluster setup (2048 H800s with InfiniBand + NVLink). Reproducing this training setup is itself a significant barrier.
- **SWE-bench gap:** Still trails Claude-3.5-Sonnet on real-world software engineering tasks (42.0% vs 50.8%), suggesting agentic/tool-use capabilities have room to grow.
- **FP8 fragility:** The FP8 training approach requires careful engineering around quantization; errors in precision handling at this scale can silently degrade quality.
- **MoE inference complexity:** Serving a 671B MoE model requires significant infrastructure — routing logic, expert sharding across many GPUs — that a dense 70B model doesn't need.
- **Post-training costs excluded:** The reported training costs cover official training only; ablations, earlier experiments, and failed runs are not counted.

## Why It Matters

DeepSeek-V3 is a landmark moment for open-source LLMs. It demonstrates that a non-US lab can train a model that competes with GPT-4o-class systems at a fraction of the assumed cost, and release it openly. The $5.576M total training cost will recalibrate industry assumptions about what frontier AI development requires.

The architectural choices — MLA, auxiliary-loss-free balancing, MTP — are each independently valuable innovations that are likely to influence the next generation of open and closed models. The FP8 training validation at 671B scale opens the door to even cheaper future training runs as FP8-capable hardware proliferates.

Perhaps most importantly, it demonstrates that careful co-design of algorithms + systems + hardware can substitute for raw compute spend — a lesson directly relevant to every lab not named Google, Microsoft, or Meta.

## See Also

[[DeepSeek-V2]] · [[Mixture of Experts]] · [[Multi-Head Latent Attention]] · [[Speculative Decoding]] · [[GRPO|Group Relative Policy Optimization]] · [[Transformer]] · [[FP8 Training]] · [[Pipeline Parallelism]]
