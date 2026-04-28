---
title: "NVFP4"
tags: [quantization, training, hardware, nvidia, precision]
tldr: "NVIDIA's 4-bit floating point training format. E2M1 elements with 16-element micro-block scaling. 3× peak throughput vs BF16 on GB300/B200 hardware. Requires a per-layer mixed-precision recipe to stay within 1% of BF16 quality."
theme: synthesis
---

# NVFP4

NVFP4 is NVIDIA's proprietary 4-bit floating point format designed for **training** (not just post-training quantization). The element format is **E2M1**: 2 exponent bits, 1 mantissa bit, giving 4-bit total with 16 representable values per element. That's an extremely narrow dynamic range — 4-bit integers have 16 values, but FP4 allocates some of those to representing a wider (if coarser) real-number range. The key to making 4-bit training accurate is **fine-grained scaling**. Every 16 consecutive elements share one E2M1 block scale factor. Every block's scale factor shares one global FP32 scale. This 2D scaling structure (element → block → global) means values get effectively represented at higher precision than a naive 4-bit integer scheme. On top of this: Random Hadamard Transforms (RHTs) are applied to weight gradient inputs before quantization to rotate the basis and suppress outliers. Stochastic rounding is used on gradients to prevent systematic underflow. The result: **3× higher peak throughput** on GB300/B200 hardware versus BF16 training. But not every layer survives 4-bit equally. QKV projections and Mamba output projections are numerically sensitive — they can flush to zero (up to 40% of activations on Nano) when quantized to NVFP4. The fix is a mixed-precision recipe: keep those layers in BF16 or MXFP8, put the rest in NVFP4. With the correct recipe, [[Nemotron-3]] achieves < 1% relative loss gap versus a full BF16 training run. The loss gap shrinks with model size — a known quantization property.

## Where it appears

- **[[Nemotron-3]]** — used throughout training (both Nano and Super); also used in post-training quantization (NVFP4 PTQ via AutoQuantize) for inference deployment

## Why it matters

- **3× throughput is a different class of scale.** Training at NVFP4 is not a marginal improvement — it fundamentally changes what you can do with a given cluster. The same 25T-token training run that takes a certain number of GPU-hours in BF16 takes one-third as many in NVFP4, or trains 3× longer on the same budget.
- **It shows that low-bit training is production-ready.** The conventional wisdom was that training below BF16 requires heroic engineering and results in degraded models. Nemotron-3's < 1% loss gap with NVFP4 training refutes this. The fine-grained micro-block scaling and per-layer sensitivity analysis are the specific techniques that made it work.
- **The per-layer sensitivity analysis is the critical engineering insight.** Not all layers tolerate 4-bit equally. QKV projections and Mamba output projections are sensitive because they directly gate information flow. Knowing which layers to keep in higher precision — and why — is what separates a successful low-bit training recipe from a failed one.

---

*Related: [[Nemotron-3]] · [[Speculative Decoding]] · [[Mamba]]*
