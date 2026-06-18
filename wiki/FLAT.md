---
created: "2026-06-11"
title: "FLAT"
aliases: ["FLAT", "Fast Linear Attention Hardware"]
year: 2025
tags: [hardware, linear-attention, acceleration, stub]
tldr: "Hardware-aware implementation of linear attention that maps the recurrent form to SRAM-resident streaming, avoiding HBM bandwidth bottlenecks."
---

## TL;DR
FLAT (Fast Linear Attention) redesigns the compute schedule for linear attention so the recurrent form runs in SRAM rather than bouncing through HBM, achieving GPU/accelerator bandwidth utilization comparable to FlashAttention.

## See Also
[[FlashAttention]] · [[Hardware-Aware Scan]] · [[Linear Attention]] · [[State Space Models]]
