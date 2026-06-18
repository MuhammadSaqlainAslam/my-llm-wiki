---
created: "2026-05-12"
title: "Near-Memory Computing"
tags: [hardware, pim, dram, exotic]
tldr: "Place compute logic *next to* memory rather than inside it (e.g., processing-in-memory DRAM stacks). A more practical compromise than in-memory analog computing — keeps digital precision while reducing the distance data has to travel."
---

# Near-Memory Computing

Near-memory computing puts a small digital compute unit on the same package, or even the same die-stack, as DRAM. Examples: Samsung's HBM-PIM, UPMEM's processing-in-DRAM. The compute is still digital (so you keep deterministic precision) but the operands don't need to traverse the long board-level memory bus to reach the accelerator. This addresses the same data-movement bottleneck as in-memory computing without the analog-conversion penalties, at the cost of more limited functionality (the in-DRAM compute units are simple). Active deployment in production is still rare but the technique is well-studied for memory-bandwidth-bound workloads.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[In-memory computing]] · [[HBM]]*
