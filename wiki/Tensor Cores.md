---
created: "2026-05-12"
title: "Tensor Cores"
tags: [hardware, gpu, nvidia, matmul, low-precision]
tldr: "Specialized matrix-multiply units inside modern NVIDIA GPUs (and the equivalent AMD Matrix Cores). Each tensor core executes a small dense matmul (e.g., $4 \times 4 \times 4$ FP16) per clock per core. The unit responsible for the 5–10× throughput jump from V100 to H100 to B200."
---

# Tensor Cores

Before tensor cores, a GPU's CUDA cores did matmul one fused-multiply-add at a time. Tensor cores group many MACs into a single hardware instruction that produces an entire small matrix product per clock. On an H100 each of the 528 tensor cores executes a $16 \times 16 \times 16$ FP16 matmul per clock; on B200 with [[NVFP4]] the throughput per core doubles again. The catch: tensor cores only see speedup if your code maps to their tile sizes and precision modes — which is why libraries like cuBLAS, CUTLASS, FlashAttention, and Triton matter so much. They do the tile choreography that keeps the tensor cores fed.

## Where it appears

- [[GPU]]
- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[NVFP4]] · [[Kernel fusion]]*
