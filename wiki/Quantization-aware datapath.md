---
created: "2026-05-12"
title: "Quantization-Aware Datapath"
tags: [hardware, quantization, int8, fp8, nvfp4]
tldr: "Silicon paths that natively execute INT8, FP8, [[NVFP4]] etc. without padding back up to FP16/BF16. The hardware support that makes low-precision inference and training actually fast."
---

# Quantization-Aware Datapath

Software quantization is easy — pick a scale factor, round, store INT8. Getting *speedup* from it is harder: if your hardware multiplies INT8 inputs but accumulates and stores in FP16, you only save bandwidth, not compute. A **quantization-aware datapath** is a hardware multiplier that consumes INT8/FP8/[[NVFP4]] operands and accumulates in a slightly higher-precision register (typically INT32 or FP32) — keeping the full pipeline at low precision until a final dequantization step. Modern tensor cores (H100 INT8 + FP8, B200 NVFP4) include exactly these datapaths and that is why low-precision inference gives the throughput boost it does. Without the matching silicon, low-precision is a memory-bandwidth optimization only, not a compute one.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]
- [[NVFP4]]

---

*Related: [[Tensor Cores]] · [[GPU]]*
