---
created: "2026-05-12"
title: "NPU (Neural Processing Unit)"
tags: [hardware, npu, edge, mobile, inference]
tldr: "Umbrella term for inference-focused accelerators in mobile and edge devices (Apple Neural Engine, Qualcomm Hexagon, Intel/AMD CPU NPUs). Optimized for INT8/FP16 throughput and energy efficiency."
---

# NPU (Neural Processing Unit)

An NPU is the on-chip neural-network accelerator inside modern phones, laptops, and edge devices. The design priorities are inverted from datacenter GPUs: instead of maximum FLOPS, the goal is maximum perf-per-watt within a tiny power envelope (≤ 5 W typical). NPUs use heavy quantization (INT8, sometimes INT4), small static dataflows, and tight integration with the surrounding SoC. They aren't general-purpose — most NPUs only run a fixed set of operator types — but for deployed inference workloads (camera ML, on-device LLMs, speech) they offer 10–100× better energy efficiency than the same task on a CPU.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[GPU]] · [[ASIC]] · [[NVFP4]]*
