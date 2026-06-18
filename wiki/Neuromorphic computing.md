---
created: "2026-05-12"
title: "Neuromorphic Computing"
tags: [hardware, spiking, exotic, energy]
tldr: "Chips designed around spiking-neuron-style computation rather than dense matmul (Loihi, TrueNorth, SpiNNaker). Promising energy efficiency on event-driven workloads; niche for current LLM-style dense models."
---

# Neuromorphic Computing

Neuromorphic chips reject the dense-matmul model of computation in favor of event-driven, sparse, asynchronous communication patterns inspired by biological neurons. Each "neuron" on the chip integrates incoming spikes and emits its own spike when it crosses a threshold. The advantage is energy: a chip that only fires when something interesting happens uses orders of magnitude less power than one that does dense matmul on every tick. The disadvantage for LLMs: modern Transformers and SSMs are intrinsically dense and synchronous, so they don't naturally map onto neuromorphic substrates. Active research area for event-driven sensing (neuromorphic vision, audio), still niche for dense neural network workloads.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[In-memory computing]]*
