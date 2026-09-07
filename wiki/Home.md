# LLM Wiki

AI research knowledge base — papers, concepts & intuitions. Artificial Intelligence Research Center, Hon Hai Research Institute.

---

## Foundations

| Article | One Line |
|---|---|
| [[Transformer]] | Self-attention replaces recurrence. Any token attends to any other in O(1). |
| [[LLaMA 2]] | Meta's open RLHF-aligned 7B–70B models. GQA at scale. The base for most community finetunes. |
| [[Mamba]] | Selective state spaces. Linear-time sequence modeling that matches Transformer quality. |

## Efficient Sequence Modeling

| Article | One Line |
|---|---|
| [[S4]] | First SSM to solve 16K-step tasks. HiPPO matrix + Cauchy kernel. The theoretical root of Mamba. |
| [[RWKV]] | Linear attention as an RNN. Trains like a Transformer, decodes with O(1) memory. Scaled to 14B. |
| [[RetNet]] | Three computation modes (parallel/recurrent/chunkwise). 15.6× throughput over Transformer at inference. |
| [[Transformers Are SSMs]] | Attention and selective SSMs are the same structured-matrix object. SSD framework gives Mamba-2: 2–8× faster, matmul-friendly. |
| [[xLSTM]] | LSTM rehab — exponential gating + matrix memory. Competitive with Transformer at 1B+ scale, O(1) inference. |
| [[Griffin]] | Google DeepMind hybrid: RG-LRU + local attention. Matches LLaMA-2 at 7× fewer training tokens. |

## Scaling

| Article | One Line |
|---|---|
| [[Mixture-of-Experts]] | Decouple parameters from compute. Route tokens to specialized FFN subnetworks. |

## Hardware & Systems

| Article | One Line |
|---|---|
| [[Flash Attention]] | IO-aware exact attention: tiles into SRAM, never writes T×T matrix to HBM. 2–4× wall-clock speedup. |
| [[FlashAttention-2]] | Better warp scheduling pushes GPU to 50–73% peak FLOPS. 225 TFLOPS/s per A100 training GPT models. |
| [[Hardware Acceleration for Neural Networks]] | Survey of GPUs, TPUs, NPUs, FPGAs, ASICs, LPUs, in/near-memory. The bottleneck is bandwidth, not FLOPS. |

## Inference Optimization

| Article | One Line |
|---|---|
| [[Medusa]] | K extra decoding heads + tree attention. 2.2× lossless speedup. No separate draft model needed. |
| [[EAGLE]] | Feature-level draft model. 3–3.5× lossless speedup on LLaMA-2-Chat 70B — best in the speculative family. |
| [[KV Cache Optimization]] | Five families: eviction, compression, hybrid memory, novel attention, combination. No universal winner — pick by workload. |
| [[Speculative Decoding]] | Fast draft proposes K tokens; large model verifies all K in one pass. ~2× throughput on long generations. |

## Modern Systems

| Article | One Line |
|---|---|
| [[Nemotron-3]] | Hybrid Mamba-Transformer-MoE. 3x throughput over pure Transformer MoE at same quality. |
| [[DeepSeek_V4]] | CSA + HCA compressed attention cuts KV cache 10× at 1M tokens. MoE + Muon optimizer. SOTA open model. |

## Looped Transformers & Recurrent Depth

| Article | One Line |
|---|---|
| [[Recurrent-Depth-Reasoning\|Huginn]] | Loop a recurrent core to arbitrary depth at inference time — a reasoning dial with no CoT data needed. |
| [[LOTUS]] | Loop + supervise parallel latents on gold CoT tokens. First latent-CoT to match explicit CoT at 3B scale. |
| [[SMELT]] | Loop the middle half of an MoE Transformer's layers under strict compute parity. Saves 6.8–18% training FLOPs. |

---

## Concept Map

```
Transformer (2017, ~100K citations)
│  Self-attention; O(n²) cost; KV cache grows linearly with context
│
│  ── RNN REVIVAL CLUSTER ─────────────────────────────────────────
├── Problem: O(n²) recurrence / attention too expensive
│   ├──► S4 (2022, ~3K)
│   │      HiPPO matrix + Cauchy kernel; O(L log L) train, O(1) decode
│   │      First to solve Path-X (16K steps)
│   ├──► RWKV (2023, ~2K)
│   │      WKV linear attention; O(1) RNN inference; scales to 14B
│   ├──► RetNet (2023, ~1K)
│   │      Retention: parallel / recurrent / chunkwise; 15.6× throughput
│   ├──► Griffin (2024, ~1K)
│   │      RG-LRU + local attention; matches LLaMA-2 at 7× fewer tokens
│   ├──► Mamba (2024)
│   │      Selective SSM (input-dependent B,C,Δ); 5× inference throughput
│   └──► xLSTM (2024)
│          Exponential gating + d×d matrix memory; O(1) inference
│
│  ── UNIFIED THEORY NODE ─────────────────────────────────────────
├── Insight: attention and SSMs are the same object
│   └──► Transformers Are SSMs (2024)
│          SSD framework: both are structured semiseparable matrices
│          ⟹ Mamba-2: 2–8× faster, matmul-friendly, TP/SP-compatible
│
│  ── HARDWARE & SYSTEMS CLUSTER ──────────────────────────────────
├── Problem: attention kernel bottlenecked by HBM bandwidth
│   ├──► FlashAttention (2022, ~8K)
│   │      IO-aware tiling into SRAM; 2–4× speedup, zero approximation
│   │      Key tricks: tiling, online softmax, recomputation
│   ├──► FlashAttention-2 (2023, ~4K)
│   │      Fix warp scheduling → 50–73% GPU utilization (vs 25–40%)
│   └──► Hardware Acceleration Survey (2025)
│          GPU / TPU / NPU / FPGA / ASIC / LPU / in-/near-memory
│          Binding constraint: memory bandwidth, not FLOPS
│
│  ── INFERENCE OPTIMIZATION CLUSTER ──────────────────────────────
├── Problem: autoregressive decode is memory-bandwidth bound
│   ├──► Speculative Decoding
│   │      Draft K tokens; verify all K in one parallel target pass
│   ├──► Medusa (2024, ~1.5K)
│   │      K frozen MLP heads + tree attention; 2.2–2.8× lossless
│   ├──► EAGLE (2024, ~1.2K)
│   │      Feature-level drafting + one-step token advance; 3–3.5×
│   └──► KV Cache Optimization (2026)
│          5 families: eviction, compression, hybrid memory,
│          novel attention, combination — choose by workload
│
├── Problem: Parameters tied to compute
│   └──► Mixture-of-Experts: sparse routing, constant FLOPs/token
│          Mixtral: top-2 of 8; beats LLaMA-2 70B at 5× fewer active params
│
├── Problem: Open RLHF-aligned model needed
│   └──► LLaMA 2 (2023, ~10K)
│          7B–70B + GQA + full RLHF recipe; democratized finetuning
│
└── Problem: Combine all insights into a production system
    ├──► Nemotron-3 (2025)
    │      Mamba-2 + sparse attention + LatentMoE + NVFP4 + MTP + RLVR
    │      7.5× throughput over Transformer MoE at 1M context
    └──► DeepSeek-V4 (2026)
           CSA + HCA compress KV cache 10× at 1M tokens
           + mHC residual + Muon optimizer + on-policy distillation
```

```
Sub-concepts by cluster:

RNN Revival:
  [[HiPPO matrix]]           optimal state-matrix A for long-range memory (S4)
  [[Cauchy kernel]]          O(L log²L) structured computation (S4)
  [[Retention mechanism]]    RetNet's parallel/recurrent/chunkwise layer
  [[RG-LRU]]                 Griffin's diagonal recurrence with decay gate
  [[Selective State Space Model]]  Mamba's input-dependent B,C,Δ (S6 cell)
  [[Semiseparable matrix]]   the unifying structure (Transformers Are SSMs)

Hardware & Systems:
  [[HBM]]        stacked DRAM; ~3–8 TB/s; the bottleneck
  [[SRAM]]       on-chip cache; 10–100× faster; where hot data must live
  [[Tiling]]     partition matmul into SRAM-sized blocks (FlashAttention)
  [[Kernel fusion]]  merge ops to avoid HBM round-trips (Mamba scan)
  [[NVFP4]]      4-bit float; 3× peak throughput vs BF16 (Nemotron-3)

Inference Optimization:
  [[Tree attention]]          verify exponential candidate tree in one pass
  [[Feature-level drafting]]  draft at hidden-state level, not token logits
  [[Lossless speedup]]        exact target distribution preserved
  [[KV Cache]]                the O(sequence) memory bottleneck
  [[Paged attention]]         16-token pages; vLLM's foundation
  [[GQA]]                     group Q heads to share KV; 16× cache reduction
  [[MQA]]                     all Q heads share one KV; 32× cache reduction

Modern Synthesis:
  [[LatentMoE]]               route in d→ℓ latent; cuts all-to-all by d/ℓ
  [[Manifold-Constrained Hyper-Connections]]  doubly-stochastic residual
  [[Muon Optimizer]]          orthogonalize gradients via Newton-Schulz
  [[GRPO]]                    group-relative RL, no value function
  [[On-Policy Distillation]]  unify specialist teachers into one student
```

---

## Reading Order

If you're new: **Transformer → LLaMA 2 → Mixture-of-Experts → Mamba → Nemotron-3**

If you care about efficiency: **S4 → Mamba → Transformers Are SSMs → xLSTM → RWKV → RetNet → Griffin**

If you're deploying: **Flash Attention → FlashAttention-2 → KV Cache Optimization → Medusa → EAGLE → Speculative Decoding**

If you care about silicon: **Hardware Acceleration → Flash Attention → FlashAttention-2 → Hardware-Aware Scan → KV Cache Optimization**

If you want the linear-RNN lineage: **S4 → RWKV → RetNet → Griffin → Mamba → Transformers Are SSMs**
