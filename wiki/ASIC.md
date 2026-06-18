---
created: "2026-05-12"
title: "ASIC (Application-Specific Integrated Circuit)"
tags: [hardware, asic, inference, fixed-function]
tldr: "Fixed-function silicon designed for one workload. Best perf-per-watt of any approach but inflexible and expensive to design (millions to tens of millions of dollars per tape-out). The class includes inference engines like AWS Inferentia, Google's TPU, and dedicated LLM-serving chips."
---

# ASIC (Application-Specific Integrated Circuit)

An ASIC is silicon designed and fabricated for one specific computation. Once taped out, the function is frozen — you cannot reprogram an ASIC the way you can an [[FPGA]] or a [[GPU]]. The reward for that loss of flexibility is the best perf-per-watt of any digital design: every transistor on the die is dedicated to the workload. The cost is enormous design and verification effort plus a one-time tape-out fee in the millions of dollars, so ASICs only pay off at high volumes (datacenter inference) or extreme efficiency requirements (edge devices). Most modern "AI accelerators" — Inferentia, Trainium, Tenstorrent, Cerebras WSE, [[LPU]]s like Groq — are ASICs.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[FPGA]] · [[GPU]] · [[TPU]] · [[LPU]]*
