---
title: "Hardware-Aware Scan"
tags: [mamba, gpu, efficiency, kernel-fusion, sram]
tldr: "Kernel fusion that keeps the Mamba SSM recurrence entirely in fast SRAM, never materializing the expanded state in HBM. 40× faster than naive PyTorch, faster than FlashAttention-2 at sequence lengths > 2K."
theme: efficiency
---

# Hardware-Aware Scan

Making [[Mamba]]'s [[State Space Model]] parameters input-dependent (B, C, Δ as functions of $x$) breaks the convolution formulation — you can't precompute the kernel because it changes at every step. You're back to sequential recurrence, and the naive approach is catastrophic: to process a batch of sequences you'd need to materialize the full expanded SSM state of shape $(B, L, D, N)$ in GPU memory, where $N \approx 16$ and $D$ is the model dimension. That's $N$ times the size of the input — easily 16× the memory you'd expect. On a GPU, the bottleneck is **HBM bandwidth** (high-bandwidth memory, the main GPU DRAM — fast by CPU standards, but slow relative to on-chip SRAM). Every read and write to HBM has latency; if you materialize the state and repeatedly read/write it, you spend most of your time on data movement, not computation. The fix: **never let the expanded state touch HBM**. Load $\Delta$, $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$ from HBM into SRAM (fast on-chip cache), perform the full discretization and recurrence entirely in SRAM, then write only the final output $(B, L, D)$ back to HBM. All intermediate states are born and die in SRAM. This is **kernel fusion** — all operations are merged into a single GPU kernel. For backpropagation, instead of saving intermediate states (which would require HBM), the backward pass recomputes them by re-loading the inputs. Memory usage matches FlashAttention-2.

## Where it appears

- **[[Mamba]]** — the hardware-aware scan is the specific algorithmic contribution that makes selective SSMs practical; without it, selectivity would come at prohibitive memory cost
- **[[Nemotron-3]]** — inherits this through Mamba-2, which extends the same principle to a more parallelizable recurrence formulation

## Why it matters

- **40× speedup over naive PyTorch.** A simple PyTorch implementation of the selective scan materializes the full $(B, L, D, N)$ state tensor and is bottlenecked by HBM reads/writes. The fused kernel avoids this entirely — every intermediate value lives in SRAM and is never written to main memory.
- **Faster than FlashAttention-2 at long sequences.** At sequence lengths > 2K tokens, the hardware-aware scan outperforms FlashAttention-2 in wall-clock time. This is the regime where Transformers are most expensive (O(n²) attention), which is precisely where the scan's advantage matters most.
- **It's the template for memory-efficient deep learning.** The same principle — fuse operations, compute in SRAM, recompute during backward instead of storing — underlies FlashAttention, Flash-Decoding, and other modern GPU kernels. Understanding this pattern is understanding why "technically equivalent but memory-efficient" implementations can be 10–100× faster in practice.

---

*Related: [[Mamba]] · [[State Space Model]] · [[Nemotron-3]]*
