---
created: "2026-05-12"
title: "Systolic Array"
tags: [hardware, tpu, dataflow, matrix]
tldr: "A 2D grid of MAC units that pipelines operands across the grid — operand A flows left-to-right, operand B flows top-to-bottom, partial sums accumulate inside each cell. The dataflow primitive at the heart of [[TPU]]s and many ASIC matmul engines."
---

# Systolic Array

A systolic array maps matrix multiplication onto a 2D grid of small multiply-accumulate cells. At each clock tick: input matrix $A$'s row elements stream right, input matrix $B$'s column elements stream down, and each cell at position $(i, j)$ multiplies the values it sees and accumulates into its local register. After enough clocks the registers contain the elements of the product matrix. The pattern was named by H.T. Kung in the 1970s by analogy with blood flowing through a heart. Strengths: near-100% MAC utilization, extremely simple wiring, scales gracefully to large dimensions. Weakness: assumes the inner kernel is a dense matmul; rigid for anything else. The structural primitive of every TPU and many inference ASICs.

## Where it appears

- [[TPU]]
- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[Tensor Cores]] · [[ASIC]]*
