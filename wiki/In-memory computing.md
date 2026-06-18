---
created: "2026-05-12"
title: "In-Memory Computing"
tags: [hardware, analog, crossbar, exotic]
tldr: "Perform multiply-accumulate inside the memory cells themselves, typically using analog crossbar arrays. Eliminates data movement for matmul-heavy layers; ADC/DAC and device-noise overheads currently erase 50–80% of the energy gain at the system level."
---

# In-Memory Computing

In conventional computing you move data from memory to a separate compute unit. **In-memory computing** flips that: the memory cells *are* the computers. An analog crossbar implements a matrix-vector product directly — each row holds an analog weight (programmed into a memristor or PCM cell), an input voltage drives down a column, and the current that exits each row is the dot product of that row with the input vector. Theoretically extremely energy-efficient (no moving bytes around) but practically constrained by analog-to-digital and digital-to-analog conversion overhead at the periphery, drift and noise of the memory devices, and limited model fraction that maps cleanly to crossbars. Active research direction; not yet competitive with digital accelerators for end-to-end LLM inference.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[Near-memory computing]] · [[Neuromorphic computing]]*
