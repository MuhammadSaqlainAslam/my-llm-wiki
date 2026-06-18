---
created: "2026-05-12"
title: "FPGA (Field-Programmable Gate Array)"
aliases: ["FPGA-LLM"]
tags: [hardware, fpga, reconfigurable]
tldr: "Reconfigurable silicon — you compile a circuit description down to bit-level routing instead of compiling instructions. Lower density than ASICs but flexible; useful for niche or rapidly changing workloads."
---

# FPGA (Field-Programmable Gate Array)

An FPGA is a sea of small lookup tables and register elements wired together by a configurable interconnect. You don't write programs that execute on it; you write a hardware description (Verilog, HLS C++) that compiles down to a circuit, which the FPGA then *becomes*. For neural networks this means you can build a datapath custom-fit to your exact model — say, an INT4 GEMM with a particular sparsity pattern — getting much better efficiency than a general-purpose GPU on that workload. The downside is density: FPGAs are typically 10–50× less power-efficient than fixed-function ASICs. Best fit: low-volume, fast-changing, or latency-critical applications where ASIC tape-out cost can't be amortized.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[ASIC]] · [[GPU]]*
