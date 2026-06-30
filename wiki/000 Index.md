---
created: "2026-04-28"
title: "LLM Wiki Index"
tags: [index, meta]
tldr: "All papers, grouped by theme, with a concept glossary and a paragraph on how everything connects."
---

# LLM Wiki Index

> Every note in one place. Start here if you want the big picture.

---

## Theme I — Foundations

The papers that started everything. Everything else in this wiki is either built on them or reacting to them.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Transformer]] | Attention Is All You Need | 2017 | Self-attention replaces recurrence; any two tokens connect in one step, enabling parallel training and O(1) path length between positions. |
| [[LLaMA 2]] | Llama 2: Open Foundation and Fine-Tuned Chat Models | 2023 | 7B–70B open models with full RLHF pipeline details. Introduces GQA for efficient KV cache. First open model competitive with ChatGPT on helpfulness; became the base for thousands of community finetunes. |

**Tags:** `foundational` `attention` `architecture` `parallelism` `rlhf` `open-source` `gqa`

---

## Theme II — Efficient Sequence Modeling

The O(n²) attention cost is fine for short sequences. For long ones it breaks. These papers replace attention with something cheaper, or generalize it.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[S4]] | Efficiently Modeling Long Sequences with Structured State Spaces | 2022 | First SSM to solve long-range benchmark tasks. Low-rank + normal A matrix → Cauchy kernel → O(L log L) convolution at training, O(1) recurrence at inference. Solves Path-X (16K steps) where all prior models fail. |
| [[RWKV]] | RWKV: Reinventing RNNs for the Transformer Era | 2023 | Linear attention with exponential decay, formulated as either Transformer (parallel training) or RNN (O(1) inference). Scales to 14B parameters — largest dense RNN ever. |
| [[RetNet]] | Retentive Network: A Successor to Transformer for Large Language Models | 2023 | Retention mechanism with 3 computation modes: parallel (train), recurrent (O(1) decode), chunkwise (long seq). 3.4× lower memory, 15.6× higher throughput than Transformer at 8K. |
| [[Mamba]] | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2024 | Make SSM parameters input-dependent so the model selectively compresses context. Hardware-aware kernel fusion makes it practical. 5× inference throughput over Transformers. |
| [[Transformers Are SSMs]] | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | 2024 | Attention and selective SSMs are two faces of the same structured semiseparable matrix. The SSD framework unlocks Mamba-2: 2–8× faster than Mamba, matmul-friendly, TP-friendly. |
| [[xLSTM]] | xLSTM: Extended Long Short-Term Memory | 2024 | Modernize the 1997 LSTM with exponential gating + matrix memory + no hidden-to-hidden recurrence. Matches Transformer/Mamba quality at 1B+ scale, keeps O(1)-per-step inference. |
| [[Griffin]] | Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models | 2024 | Google DeepMind's RG-LRU + local attention hybrid. Griffin-14B matches LLaMA-2 on 7× fewer training tokens. Hawk (pure recurrence) beats Mamba at 3B. |
| [[Mamba3\|Mamba-3]] | Mamba-3: Improved Sequence Modeling using State Space Principles | 2026 | Three principled SSM improvements — exponential-trapezoidal discretization, complex-valued states, MIMO formulation — +1.9pp over Mamba-2 at 1.5B params; matches Mamba-2 perplexity at half the state size. |
| [[Gated-DeltaNet-2\|Gated DeltaNet-2]] | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention | 2026 | Separate channel-wise erase and write gates replace the tied scalar gate in Gated DeltaNet/KDA; strongest among Mamba-2/3 and DeltaNet-family variants, especially on long-context retrieval. |

**Tags:** `ssm` `efficiency` `linear-time` `recurrence` `selectivity` `lstm` `gating` `mamba-2` `ssd` `rnn` `linear-attention` `retention`

---

## Theme III — Scaling / Parameter Efficiency

Parameters and compute don't have to scale together. These papers decouple them.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Mixture-of-Experts]] | Switch Transformers / Mixtral of Experts | 2022 | Route each token to a sparse subset of expert FFNs; parameters scale cheaply while per-token compute stays constant. More parameters, same FLOPs. |
| [[LatentMoE]] | Nemotron 3 Technical Report | 2025 | Project tokens $d \rightarrow \ell$ before routing; cuts all-to-all communication by $d/\ell$ and enables 4× more experts at the same inference cost. |
| [[State-Space-Models\|State-Space Models]] | (Concept note) | — | Linear recurrence over a fixed-size hidden state; linear-time in sequence length, O(1) inference memory. Foundation for S4, Mamba, and all hybrid SSM architectures. |

**Tags:** `scaling` `moe` `routing` `efficiency` `sparse`

---

## Theme IV — Modern Synthesis

What happens when you combine the three themes above into a single production system.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Nemotron-3]] | Nemotron 3: Efficient and Open Intelligence | 2025 | Hybrid Mamba-2 + sparse attention + LatentMoE, trained at NVFP4. 120B/12B active. 7.5× throughput over Qwen3.5-122B at 1M context with competitive accuracy. |
| [[DeepSeek_V4]] | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | 2026 | CSA + HCA compressed attention yields 10× KV cache reduction at 1M tokens. mHC residual connections + Muon optimizer. 1.6T/49B active. SOTA open model. |

**Tags:** `hybrid` `moe` `mamba` `production` `inference` `quantization` `rl` `agentic`

---

## Hardware & Systems

The silicon and memory systems the algorithms have to run on. Knowing the hardware constraints explains *why* every other paper in this wiki looks the way it does.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Flash Attention]] | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | IO-aware exact attention: tiles into SRAM, never writes T×T matrix to HBM. 2–4× wall-clock speedup, zero approximation. The template every serious attention kernel copies. |
| [[FlashAttention-2]] | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | Fixes v1's warp scheduling to push GPU utilization from 25–40% to 50–73% of peak FLOPS. Reaches 225 TFLOPS/s per A100 training GPT models. |
| [[Hardware Acceleration for Neural Networks]] | Xu, Banerjee & Gupta — A Comprehensive Survey | 2025 | Survey of every chip family running NNs (GPU, TPU, NPU, FPGA, ASIC, LPU, in/near-memory, neuromorphic). The recurring theme: peak FLOPS doesn't matter — memory bandwidth and KV-cache I/O dominate end-to-end performance. |

**Tags:** `hardware` `gpu` `tpu` `npu` `fpga` `asic` `lpu` `in-memory` `kv-cache` `io-awareness` `kernel-fusion`

---

## Inference Optimization

Once a model is trained, the cost is moving bytes. These papers attack inference latency, throughput, and KV-cache memory directly.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Medusa]] | MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | 2024 | K extra decoding heads on the frozen LLM predict K future tokens in parallel. Tree attention verifies all candidate continuations in one forward pass. No separate draft model needed. 2.2–2.8× speedup. |
| [[EAGLE]] | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | 2024 | Draft at the hidden-state (feature) level, not token level. One lightweight autoregressive head predicts features given the actual next token. 3–3.5× lossless speedup on LLaMA-2-Chat 70B. |
| [[Speculative Decoding]] | (Concept note) | — | Fast draft model proposes K tokens; large model verifies all K in one parallel pass. Expected accepted tokens per call = K × acceptance-rate. With MTP heads as drafts, ~2× throughput. |
| [[KV Cache Optimization]] | Xu, Khaira & Singh — KV Cache Optimization Strategies for Scalable and Efficient LLM Inference | 2026 | Five families of KV-cache optimization (eviction, compression, hybrid memory, novel attention, combination), mapped to seven production scenarios. No single technique dominates — choose by workload. |
| [[Quantization]] | (Concept note) | — | Reduce weight/activation precision (FP8, NVFP4) to shrink memory and increase throughput; requires component-specific handling for SSM states and attention layers. |

**Tags:** `inference` `kv-cache` `throughput` `eviction` `compression` `paged-attention` `speculative-decoding` `medusa` `eagle` `draft-model`

---

## High-Impact Papers (1000+ Citations)

Ranked by approximate citation count as of 2025. These are the most-referenced papers in the ecosystem covered by this wiki.

| Rank | Note | Year | ~Citations | One-Line Impact |
|---|---|---|---|---|
| 1 | [[Transformer]] | 2017 | ~100,000 | Invented self-attention; made every LLM possible |
| 2 | [[LLaMA 2]] | 2023 | ~10,000 | First open RLHF-aligned model competitive with ChatGPT; democratized finetuning |
| 3 | [[Flash Attention]] | 2022 | ~8,000 | IO-aware exact attention; enabled 32K+ context training in practice |
| 4 | [[FlashAttention-2]] | 2023 | ~4,000 | Pushed attention to 50–73% of GPU peak FLOPS; the default attention kernel today |
| 5 | [[S4]] | 2022 | ~3,000 | First SSM to solve 16K-step tasks; theoretical foundation for Mamba |
| 6 | [[RWKV]] | 2023 | ~2,000 | Proved linear RNNs scale to 14B; parallel training + O(1) inference |
| 7 | [[Medusa]] | 2024 | ~1,500 | Speculative decoding without a draft model; 2.2× speedup from extra heads |
| 8 | [[EAGLE]] | 2024 | ~1,200 | Feature-level drafting; 3–3.5× lossless speedup on 70B models |
| 9 | [[RetNet]] | 2023 | ~1,000 | Three computation modes; 15.6× higher inference throughput than Transformer |
| 10 | [[Griffin]] | 2024 | ~1,000 | Google DeepMind hybrid; matches LLaMA-2 at 7× fewer training tokens |

---

## Concept Glossary

Grouped by theme. Each entry links to its own wiki note with full explanation.

### Attention & Transformers

- [[Multi-head attention]] — Run h independent attention heads in parallel; each learns different relationship types between tokens.
- [[Flash Attention]] — IO-aware exact attention: tile into SRAM, never write the T×T matrix to HBM; 2–4× speedup with zero approximation.
- [[FlashAttention-2]] — Fixes v1's warp scheduling to push GPU utilization from ~25% to 50–73% of peak FLOPS.
- [[Online softmax]] — Numerically stable tile-by-tile softmax that never materializes the full attention matrix.
- [[IO-awareness]] — Design algorithms around HBM↔SRAM data movement cost, not raw FLOPs.
- [[Tiling]] — Partition a matrix computation into SRAM-sized blocks to control memory residency.
- [[Recomputation]] — Recompute attention scores in the backward pass instead of storing the T×T matrix; trades FLOPs for memory.
- [[Warp]] — 32-thread lockstep execution unit inside a GPU; inter-warp scheduling is the bottleneck FlashAttention-2 fixes.
- [[Occupancy]] — Fraction of a GPU's warps actively executing; the metric FlashAttention-2 optimizes.
- [[Thread block]] — Group of warps sharing on-chip SRAM; the work unit assigned to one streaming multiprocessor.
- [[GQA]] — Group query heads to share fewer KV heads; 16× KV cache reduction at near-zero quality cost.
- [[MQA]] — Multi-Query Attention: all query heads share one KV head; extreme cache reduction with some quality loss.
- [[Linear attention]] — Replace softmax with a kernel so Q(KᵀV) = Q(Kᵀ)V; O(Td²) instead of O(T²d).
- [[Local attention]] — Attend only to the last W tokens; O(W·T) cost, bounded memory for very long sequences.
- [[Sliding window attention]] — Keep the last W tokens in the KV cache; oldest fall off the window.
- [[Attention sinks]] — First tokens accumulate disproportionate attention weight and cannot safely be evicted.
- [[Spike-Sparse-Sink]] — Massive activations and attention sinks co-occur due to pre-norm architecture, not functional necessity; removing pre-norm decouples them (Sun, Canziani, LeCun, Zhu 2026).
- [[Compressed Sparse Attention]] — Compress m tokens → 1 KV entry via learned weighted sum, then sparse top-k block selection.
- [[Heavily Compressed Attention]] — Compress m' >> m tokens → 1 KV entry; coarser than CSA, cheaper, for very distant context.
- [[RoPE]] — Encodes position by rotating Q and K vectors in 2D planes by an angle proportional to position.
- [[XPOS]] — RetNet's relative position embedding: multiply Q and K by exponentially decaying position-dependent scalars.

### State Space Models

- [[State Space Model]] — Continuous-time linear dynamical system discretized for sequences: h_t = Āh_{t-1} + B̄x_t.
- [[S4]] — First SSM to solve long-range benchmarks; HiPPO matrix + Cauchy kernel gives O(L log L) training.
- [[Mamba]] — Selective SSM: B, C, Δ are input-dependent so the model compresses context by content, not position.
- [[RWKV]] — Linear attention with WKV exponential decay; parallel training like a Transformer, O(1) inference like an RNN.
- [[RetNet]] — Three computation modes (parallel/recurrent/chunkwise); 15.6× higher throughput vs Transformer at 8K length.
- [[Griffin]] — RG-LRU + local attention hybrid; matches LLaMA-2 at 7× fewer training tokens.
- [[xLSTM]] — LSTM modernized with exponential gating + matrix memory; competitive with Transformer at 1B+ scale.
- [[Transformers Are SSMs]] — Proves attention and selective SSMs are both structured semiseparable matrices; gives Mamba-2.
- [[HIPPO]] — Mathematical framework deriving optimal SSM state matrix A via polynomial projection of input history.
- [[HiPPO matrix]] — The specific matrix A derived from HiPPO; gives S4 principled long-range memory from initialization.
- [[Cauchy kernel]] — K_{ij} = 1/(ω_i - ζ_j) structured matrix arising from diagonalized SSM; enables O(L log²L) computation.
- [[Low-rank correction]] — A = diagonal + rank-1 decomposition enabling the DPLR structure computable via Cauchy kernel.
- [[Discretization]] — Convert ODE x' = Ax + Bu into discrete recurrence; ZOH gives Ā = exp(ΔA).
- [[Convolutional view]] — SSMs during training are a global convolution y = K̄ * u; fully parallelizable via FFT.
- [[Selective State Space Model]] — SSM whose B, C, Δ depend on the current input; the S6 cell at the heart of Mamba.
- [[Semiseparable matrix]] — Matrix whose every off-diagonal block has rank ≤ N; the unifying structure behind SSMs and attention.
- [[Multi-head SSM]] — Multiple parallel SSM heads by analogy with multi-head attention; used in Mamba-2.
- [[Retention mechanism]] — RetNet's decayed causal attention substitute supporting parallel/recurrent/chunkwise computation.
- [[Multi-scale retention]] — Multiple retention heads with different γ decay values; like multi-head attention with different time scales.
- [[Chunkwise recurrent]] — Intra-chunk parallel + inter-chunk recurrent; linear-complexity long-sequence processing.
- [[Positional decay]] — Attention weight between positions decays as γ^(i-j); RetNet and RWKV's implicit position encoding.
- [[Recurrent state]] — Fixed-size hidden state h_t that summarizes all past context for RNN-style models.
- [[Diagonal recurrence]] — Linear recurrence h_t = a_t ⊙ h_{t-1} + b_t with diagonal transition; efficient on parallel hardware.
- [[RG-LRU]] — Griffin's Real-Gated Linear Recurrent Unit: diagonal recurrence with input-dependent scalar decay gate.
- [[LRU]] — Complex-valued diagonal linear recurrence with eigenvalue initialization; theoretical precursor to RG-LRU.
- [[Exponential decay]] — Linear RNNs discount older tokens via a multiplicative scalar < 1 at each recurrent step.
- [[Time-mixing]] — RWKV sublayer for cross-time interaction via the WKV exponential decay recurrence.
- [[Channel-mixing]] — RWKV pointwise FFN-like sublayer for within-position transformation.
- [[Time shift]] — RWKV's blend of current token embedding with previous token's embedding; cheap temporal context.
- [[Receptance]] — RWKV's output gate r_t = sigmoid(W_r x̂_t) that controls how much the WKV signal contributes.
- [[LSTM]] — 1997 Hochreiter–Schmidhuber recurrent network with cell state constant-error carousel.
- [[Constant error carousel]] — Identity path through LSTM cell state; the original solution to vanishing gradients.
- [[Exponential gating]] — Replace sigmoid gates with exp(·) + running max-stabilizer; a single token can reset memory.
- [[Matrix memory]] — Replace LSTM's scalar cell with a d×d matrix updated by outer product; a learned key-value store.
- [[Covariance update rule]] — C_t = f_t C_{t-1} + i_t v_t k_tᵀ; the outer-product update at the heart of mLSTM.
- [[sLSTM]] — xLSTM scalar-memory cell with exponential gating + cross-cell memory mixing.
- [[mLSTM]] — xLSTM matrix-memory cell; fully parallelizable during training.
- [[Long Range Arena]] — 6-task benchmark suite testing long-range dependencies up to 16K steps; S4 first solved Path-X.

### Inference Optimization

- [[Speculative Decoding]] — Draft model proposes K tokens; target model verifies all K in one parallel pass; ~2× throughput.
- [[Medusa]] — K extra frozen-backbone MLP heads predict K future tokens; tree attention verifies; 2.2–2.8× lossless speedup.
- [[EAGLE]] — Feature-level autoregressive draft model; 3–3.5× lossless speedup on LLaMA-2-Chat 70B.
- [[KV Cache]] — Cached key-value tensors from past attention steps; grows linearly with sequence length.
- [[KV Cache Optimization]] — Five families of techniques (eviction, compression, hybrid memory, novel attention, combination) for bounded cache.
- [[Cache eviction]] — Drop KV-cache entries to keep memory bounded; the simplest optimization family.
- [[Cache compression]] — Keep all KV entries but shrink each via INT8/INT4/NVFP4 quantization or low-rank approximation.
- [[Hybrid memory]] — Tier KV cache across HBM / host DRAM / NVMe; page hot blocks back for decoding.
- [[Novel attention mechanisms]] — Architectural changes that structurally shrink the KV cache (GQA, MQA, CSA, HCA).
- [[Combination strategies]] — Stack two or more optimization families; the production reality for large-scale serving.
- [[Eviction policy]] — Rule deciding which KV-cache entries to drop (LRU, attention-score, attention-sinks-aware).
- [[Token budget]] — Hard cap on KV-cache size the eviction policy works within.
- [[H2O eviction]] — Drop tokens with lowest accumulated attention scores; heavy hitter oracle eviction.
- [[Paged attention]] — Store KV cache in 16-token pages like OS virtual memory; foundation of vLLM.
- [[KV offloading]] — Page cold KV blocks from HBM to host DRAM or NVMe SSD.
- [[Prefill]] — Initial parallel pass that processes the prompt and populates the KV cache.
- [[Decode phase]] — Autoregressive token-by-token generation; memory bandwidth-bound.
- [[Draft model]] — Small fast model in speculative decoding that proposes K candidate tokens.
- [[Verification step]] — Single parallel target-model pass that accepts/rejects all K draft tokens simultaneously.
- [[Draft model-free speculative decoding]] — Speculative decoding without a separate model; Medusa is the canonical example.
- [[Lookahead Decoding]] — Jacobi-iteration-based method that speculatively drafts multiple candidate continuations simultaneously.
- [[Tree attention]] — Modified attention mask encoding a tree of candidates; verifies all paths in one forward pass.
- [[Tree-based decoding]] — Organizing speculative candidates as a tree to verify exponentially more paths per pass.
- [[Medusa heads]] — K extra 2-layer MLP heads on a frozen LLM; each predicts one token further into the future.
- [[Medusa-1]] — Medusa variant where only the extra heads are fine-tuned; plug-and-play with any frozen LLM.
- [[Medusa-2]] — Medusa variant where both extra heads and backbone are jointly fine-tuned for higher quality.
- [[Typical acceptance scheme]] — Medusa's acceptance criterion based on typical set filtering instead of full speculative rejection.
- [[Lossless speedup]] — Acceleration method that produces samples from the exact target distribution.
- [[Feature-level drafting]] — Draft at the second-to-last hidden state rather than token logits; smoother prediction surface.
- [[Feature uncertainty]] — Next hidden state depends on sampled token; EAGLE resolves via one-step token advance.
- [[One-step token advance]] — EAGLE's trick: shift the actual next token into the draft input, collapsing feature uncertainty.
- [[Inference optimization]] — The family of post-training techniques reducing wall-clock time, memory, and cost of serving.
- [[Multi-Token Prediction]] — Predict 2, 3… tokens ahead simultaneously; richer training signal + free speculative drafts.

### Hardware & Systems

- [[GPU]] — Massively parallel SIMT processor; dominant DL accelerator with tensor cores + HBM.
- [[TPU]] — Google ASIC built around large systolic arrays; high matmul density per watt.
- [[NPU]] — Inference-focused accelerator for mobile/edge devices; INT8-optimized, low-power.
- [[FPGA]] — Reconfigurable silicon; compile a circuit, not a program; flexible but less dense than ASIC.
- [[ASIC]] — Fixed-function silicon for one workload; best perf/W; expensive to design.
- [[LPU]] — Language Processing Unit: ASIC engineered for low-latency LLM token generation (Groq, SambaNova).
- [[Systolic array]] — 2D MAC grid that pipelines operands across rows and columns; TPU's core compute unit.
- [[Tensor Cores]] — Specialized matmul engines inside modern NVIDIA GPUs; execute a dense matmul per clock cycle.
- [[HBM]] — Stacked DRAM next to the accelerator die; ~3–8 TB/s bandwidth; the dominant GPU memory tier.
- [[SRAM]] — On-chip cache (KB–MB); 10–100× faster than HBM; where hot data must live for peak throughput.
- [[Memory hierarchy]] — Registers → SRAM → HBM → host RAM → SSD; performance set by where data lives.
- [[Kernel fusion]] — Combine consecutive ops into one GPU kernel so intermediates stay in SRAM instead of round-tripping HBM.
- [[Operator fusion]] — Compiler-level kernel fusion (XLA, Inductor, TVM); automatic version of kernel fusion.
- [[Hardware-aware algorithms]] — Algorithms designed around the memory hierarchy and parallelism of the target chip.
- [[Hardware-Aware Scan]] — Kernel fusion keeping Mamba's SSM recurrence entirely in SRAM; avoids HBM reads for intermediates.
- [[NVFP4]] — NVIDIA 4-bit float (E2M1 + 16-element micro-block scaling); 3× peak throughput vs BF16.
- [[Pallas kernel]] — Google's JAX-based language for custom GPU/TPU kernels; analogous to Triton.
- [[Quantization-aware datapath]] — Silicon paths natively executing INT8/FP8/NVFP4 without padding to FP32.
- [[In-memory computing]] — Perform multiply-accumulate inside analog memory crossbars; research stage.
- [[Near-memory computing]] — Place compute logic next to memory (HBM-PIM, UPMEM) to reduce data movement.
- [[Neuromorphic computing]] — Spiking-neuron-inspired chips (Loihi, TrueNorth); niche, not yet competitive for LLMs.

### Training & Optimization

- [[RLHF]] — Reinforcement Learning from Human Feedback: reward model on preference pairs, then PPO to align the LLM.
- [[RLVR]] — RL with verifiable rewards (math, code, tool-use); eliminates subjective human ratings, harder to game.
- [[GRPO]] — PPO variant without a value function; normalizes advantages within a group of rollouts per prompt.
- [[Ghost Attention]] — Training trick for persistent instruction following: append system message to every turn, mask in loss.
- [[Load Balancing Loss]] — Auxiliary loss penalizing unequal token distribution across MoE experts.
- [[On-Policy Distillation]] — Student generates own rollouts and minimizes reverse KL vs. ensemble of specialist teachers.
- [[Muon Optimizer]] — Orthogonalize gradient matrix via Newton-Schulz iterations before applying; isotropic update, faster convergence.
- [[Tensor parallelism]] — Shard layer weights across GPUs on the same node; each GPU holds a column slice of each matrix.
- [[Sequence parallelism]] — Shard the sequence dimension across GPUs; pass recurrent state at chunk boundaries for SSMs.

### Architecture Components

- [[Transformer]] — Self-attention encoder-decoder; the universal backbone of modern NLP since 2017.
- [[LLaMA 2]] — Meta's 7B–70B open models with RLHF pipeline and GQA; the community finetuning base.
- [[Mixture-of-Experts]] — Sparse routing: each token activates K of N expert FFNs; decouples parameters from FLOPs.
- [[Mixtral]] — Mistral's 8-expert top-2 MoE Transformer; outperforms LLaMA-2 70B at 5× fewer active parameters.
- [[Nemotron-3]] — Mamba-2 + sparse attention + LatentMoE + NVFP4; 7.5× inference throughput at 1M context.
- [[DeepSeek_V4]] — CSA + HCA attention compression + mHC residual + Muon; 1M-token context at 27% of V3 FLOPs.
- [[LatentMoE]] — Project tokens d → ℓ before routing; cuts all-to-all communication by d/ℓ and enables more experts.
- [[Manifold-Constrained Hyper-Connections]] — Expand residual stream to n_hc × d; doubly-stochastic mixing matrix guarantees spectral norm ≤ 1.
- [[Selective State Spaces]] — Synonym for Selective State Space Model; the S6 parameterization in Mamba.
- [[LLM evaluation]] — Measuring LLM capability across standardized benchmarks (MMLU, MATH, HumanEval, RULER, SWE-bench).
- [[LLM Benchmarks]] — 2025 comparison across frontier models on knowledge, reasoning, code, long-context, and agentic tasks.

---

## Citation Intelligence

Downstream impact of the six core papers in this wiki — total citations plus the ten most-cited works they spawned. Full breakdown in [[Citation Map]].

| Paper | Total Citations | Field Impact |
|---|---|---|
| [[Mamba]] | ~4,841 | Transformative — spawned entire SSM ecosystem |
| [[Transformers Are SSMs]] | ~1,200 | High — unified theory of SSM+Transformer |
| [[xLSTM]] | ~800 | High — revived LSTM research direction |
| [[Speculative Decoding]] | ~600 | Growing — standard inference technique |
| [[KV Cache Optimization]] | ~200 | Emerging — practical inference impact |
| [[Hardware Acceleration for Neural Networks]] | ~120 | Specialized — hardware co-design |

→ [[Citation Map]] — full per-paper citation tables with top 10 citing works ranked by citation count

---

## How These Papers Connect

The **Transformer** (Vaswani et al., 2017) solved sequential training by replacing recurrence with self-attention, giving every token direct access to every other in O(1) path length — but at O(n²) compute and a KV cache that grows without bound. That constraint became the organizing force behind the next decade of research. Three enabling technologies arrived in 2022–2023 to make the Transformer actually usable at scale: [[Flash Attention]] showed that the O(n²) computation was never necessary — tile attention into SRAM and avoid the HBM round-trips entirely for 2–4× wall-clock speedup with zero approximation, enabling 32K+ context training in practice. [[S4]] showed that continuous-time state space models, initialized via the [[HIPPO|HiPPO]] polynomial projection framework, could solve long-range benchmarks (including 16K-step Path-X) where all prior architectures failed — laying the theoretical foundation for every recurrent model that followed. And [[LLaMA 2]] gave the open community the RLHF training blueprint: 7B–70B pretrained-and-chat-tuned models with full GQA scaling details and Ghost Attention for multi-turn instruction consistency, democratizing fine-tuning for thousands of downstream models.

The **linear-RNN revival** of 2023–2024 proved that recurrence, done correctly, can match Transformer quality with O(1)-per-step inference. [[RWKV]] formulated exponential time-decay attention as a pure recurrence, scaling to 14B parameters — the first linear RNN to match Transformer quality at that scale. [[RetNet]] gave the retention mechanism three equivalent computation modes (parallel for training, recurrent for O(1) decode, chunkwise for long sequences), achieving 15.6× higher throughput than Transformer at 8K context. [[Griffin]] showed that RG-LRU plus local sliding-window attention matches LLaMA-2 at 7× fewer training tokens — proving the hybrid is more data-efficient than pure attention. [[Mamba]] closed the remaining quality gap by making SSM parameters input-dependent, so the model selectively compresses context by *content* rather than position alone, paired with a hardware-aware kernel fusion pass. [[Transformers Are SSMs]] unified everything: selective SSMs and attention are both parameterizations of structured semiseparable matrices, giving Mamba-2 which is 2–8× faster than Mamba-1 while inheriting FlashAttention IO-awareness, tensor parallelism, and sequence parallelism from the Transformer toolbox. [[xLSTM]] took the third path — neither SSM nor attention but a modernized 1997 LSTM, fixed with exponential gating (a single token can override stored memory) and d×d matrix memory (cell becomes a learned key-value store) — closing the quality gap at billion-parameter scale.

**Mixture-of-Experts** (Switch Transformers / [[Mixtral]]) attacked a different axis: decouple parameter count from compute by routing each token to K of N expert FFNs — more knowledge without more FLOPs. [[Nemotron-3]] (NVIDIA, 2025) synthesizes all three themes: Mamba-2 recurrence, sparse attention, LatentMoE (routes in a compressed latent space to cut all-to-all communication by d/ℓ), NVFP4 training (3× hardware throughput over BF16), multi-token prediction (denser training signal + free speculative drafts at inference), and 21-environment simultaneous RL — into a production system delivering 7.5× throughput over comparably-sized Transformer MoEs at 1M context. [[DeepSeek_V4]] (2026) attacks the O(n²) problem from the Transformer side rather than replacing it: Compressed Sparse Attention squeezes m tokens into one KV entry then sparse-selects the top-k relevant blocks; Heavily Compressed Attention goes coarser for distant context; together they cut the KV cache 10× at 1M tokens. Manifold-Constrained Hyper-Connections expand and stabilize the residual stream via a doubly-stochastic mixing matrix (spectral norm ≤ 1 guaranteed); the Muon optimizer orthogonalizes gradient updates for faster convergence. Post-training distills specialist expert models into a unified model via on-policy rollouts.

The **inference layer** runs orthogonally to every architecture choice. [[Medusa]] attaches K frozen MLP heads to an existing LLM that predict K future tokens simultaneously — tree attention verifies all candidate continuations in one forward pass, 2.2–2.8× lossless speedup without a separate draft model. [[EAGLE]] drafts at the feature level (second-to-last hidden state), feeding the actual next token into the draft input to collapse the feature uncertainty problem — 3–3.5× lossless speedup on 70B models. These pair with [[KV Cache Optimization]] (2026), which maps five families of cache techniques to seven production workload scenarios (no single strategy dominates), and with [[FlashAttention-2]] (2023), which pushes GPU attention utilization to 50–73% of peak FLOPS by fixing warp scheduling. The [[Hardware Acceleration for Neural Networks]] survey explains the binding constraint behind every technique: LLM inference is memory-bandwidth-bound, not compute-bound. Reading the model and KV cache from HBM is the bottleneck. Every paper in this wiki — from FlashAttention's tiling to Mamba's hardware-aware scan to EAGLE's feature drafting — is answering the same question: *how do you move fewer bytes between silicon and memory per token of intelligence produced?*

---

## Benchmarks & Evaluation

Quantitative comparisons across frontier models on knowledge, long context, and agentic tasks.

| Note | Coverage | Year | TL;DR |
|---|---|---|---|
| [[LLM Benchmarks]] | DS-V4-Pro, DS-V4-Flash, K2.6, GLM-5.1, Opus-4.6, GPT-5.4, Gemini-3.1-Pro — 22 benchmarks | 2025 | Gemini leads knowledge, Opus-4.6 leads long context, agentic tasks split across GPT-5.4 / Opus-4.6 / K2.6; DS-V4-Pro is strongest open model. |
| [[Opus 4.8 Benchmarks]] | Opus 4.8 vs Opus 4.7, GPT-5.5, Gemini 3.1 Pro — 7 benchmarks | 2026 | Opus 4.8 leads SWE-bench Pro (69.2%), OSWorld (83.4%), GDPval-AA (1890 Elo); GPT-5.5 wins Terminal-Bench 2.1. |
| [[SWE-bench]] | Real GitHub issue resolution across 300 repos | 2023 | 12.3% resolved by best model at release; now exceeded 80%+ by frontier models. |
| [[SWE-bench Pro]] | Harder enterprise coding tasks, less contamination | 2025 | More realistic multi-file tasks; Opus 4.8 scores 69.2% Pass@1. |
| [[OSWorld]] | GUI agent on real desktop OS (Ubuntu/Windows/macOS) | 2024 | 369 computer tasks; frontier models reach 80%+ on verified subset. |
| [[Humanity's Last Exam]] | 2,500 expert PhD-level questions | 2025 | Designed to resist saturation; GPT-4o scored <10% at launch. |
| [[MCP-Atlas]] | LLM tool use via real MCP servers | 2026 | Tests actual MCP protocol tool invocations across diverse server types. |
| [[OSWorld-MCP]] | MCP tools integrated into computer-use agents | 2025 | Extends OSWorld with structured MCP tool access. |
| [[ToolLLM]] | 16,464 real-world APIs from RapidAPI | 2023 | Evaluates tool planning, selection, and chained API calls. |

**Tags:** `benchmarks` `evaluation` `llm-comparison`

---

## 🔧 Tool Use & Agentic Benchmarks

Evaluates LLM ability to use external tools, APIs, and computer interfaces.

| Paper | Year | What it measures |
|-------|------|-----------------|
| [[ToolLLM]] | 2023 | 16,464 real-world APIs — tool planning and chaining |
| [[OSWorld]] | 2024 | GUI desktop control via mouse/keyboard on real OS |
| [[SWE-bench Pro]] | 2025 | Enterprise agentic coding across multi-file repos |
| [[OSWorld-MCP]] | 2025 | MCP tool invocation within computer-use agents |
| [[MCP-Atlas]] | 2026 | Real MCP server tool use across diverse task types |

---

## 🧠 Reasoning & RL

Papers on reasoning, reinforcement learning from human feedback, chain-of-thought, and alignment.

| Paper | Year | Citations | arXiv |
|-------|------|-----------|-------|
| [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models\|Chain of Thought]] | 2022 | ~9,000 | [2201.11903](https://arxiv.org/abs/2201.11903) |
| [[RLHF\|InstructGPT / RLHF]] | 2022 | ~8,500 | [2203.02155](https://arxiv.org/abs/2203.02155) |
| [[ReAct Synergizing Reasoning and Acting in Language Models\|ReAct]] | 2022 | ~3,500 | [2210.03629](https://arxiv.org/abs/2210.03629) |
| [[Direct Preference Optimization Your Language Model is Secretly a Reward Model\|DPO]] | 2023 | ~4,500 | [2305.18290](https://arxiv.org/abs/2305.18290) |
| [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning\|DeepSeek-R1]] | 2025 | ~8,000 | [2501.12948](https://arxiv.org/abs/2501.12948) |
| [[Multi-Environment RLVR Training]] | 2025 | — | Technical report ([2512.20856](https://arxiv.org/abs/2512.20856)) |
| [[VinePPO]] | 2024 | — | [2410.01679](https://arxiv.org/abs/2410.01679) |

**Tags:** `reasoning` `rlhf` `alignment` `chain-of-thought` `credit-assignment`

---

## 🌍 World Models & World-Action Models

Models that learn a predictive representation of an environment, used either to plan or to train a policy "in imagination":

| Paper | Year | Citations | arXiv |
|-------|------|-----------|-------|
| [[World Models]] | 2018 | — | [1803.10122](https://arxiv.org/abs/1803.10122) |
| [[World-Model-Landscape\|The World Model Landscape (2019-2026)]] | 2026 | — | Industry report (BVP) |

*(More entries to be added as DreamerV3, IRIS, DIAMOND, and Loop-OWM are verified and added.)*

**Tags:** `world-models` `reinforcement-learning` `generative-models` `planning`

---

## 🔧 Fine-tuning & Alignment

Parameter-efficient fine-tuning and alignment techniques.

| Paper | Year | Citations | arXiv |
|-------|------|-----------|-------|
| [[LoRA Low-Rank Adaptation of Large Language Models\|LoRA]] | 2021 | ~12,000 | [2106.09685](https://arxiv.org/abs/2106.09685) |
| [[RLHF\|InstructGPT / RLHF]] | 2022 | ~8,500 | [2203.02155](https://arxiv.org/abs/2203.02155) |
| [[Direct Preference Optimization Your Language Model is Secretly a Reward Model\|DPO]] | 2023 | ~4,500 | [2305.18290](https://arxiv.org/abs/2305.18290) |
| [[Ministral 3]] | 2026 | — | [2601.08584](https://arxiv.org/abs/2601.08584) |

**Tags:** `fine-tuning` `alignment` `parameter-efficient` `rlhf` `distillation`

---

## 🌐 Frontier Model Families

Technical reports of major model families (2024–2025).

| Paper | Year | Citations | arXiv |
|-------|------|-----------|-------|
| [[The Llama 3 Herd of Models\|LLaMA 3]] | 2024 | ~15,000 | [2407.21783](https://arxiv.org/abs/2407.21783) |
| [[DeepSeek-V3 Technical Report\|DeepSeek-V3]] | 2024 | ~3,000 | [2412.19437](https://arxiv.org/abs/2412.19437) |
| [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning\|DeepSeek-R1]] | 2025 | ~8,000 | [2501.12948](https://arxiv.org/abs/2501.12948) |
| [[Qwen3 Technical Report\|Qwen3]] | 2025 | ~500 | [2505.09388](https://arxiv.org/abs/2505.09388) |
| [[Nemotron-3]] | 2025 | 14 | [2512.20856](https://arxiv.org/abs/2512.20856) |
| [[Nemotron_3_Super]] | 2026 | — | [2604.12374](https://arxiv.org/abs/2604.12374) |
| [[Nemotron_3_Ultra]] | 2026 | — | Technical report (no arXiv) |
| [[MAI-Thinking-1]] | 2026 | Microsoft | Technical report (no arXiv) |

**Tags:** `llm` `foundational` `scaling` `open-source`

---

## 👁️ Vision-Language Models

Models and techniques connecting vision encoders to large language models.

| Paper | Year | Citations | arXiv |
|-------|------|-----------|-------|
| [[MiniGPT-4]] | 2023 | — | [2304.10592](https://arxiv.org/abs/2304.10592) |
| [[LLaVA-1.5]] | 2023 | — | [2310.03744](https://arxiv.org/abs/2310.03744) |

**Tags:** `vision-language` `multimodal` `instruction-tuning`

---

## Reading Order

**If you're new:** [[Transformer]] → [[LLaMA 2]] → [[Mixture-of-Experts]] → [[Mamba]] → [[Nemotron-3]]

**If you care about efficiency:** [[S4]] → [[Mamba]] → [[Transformers Are SSMs]] → [[xLSTM]] → [[RWKV]] → [[RetNet]] → [[Griffin]]

**If you're deploying:** [[Flash Attention]] → [[FlashAttention-2]] → [[KV Cache Optimization]] → [[Medusa]] → [[EAGLE]] → [[Speculative Decoding]]

**If you want the math:** [[Transformer]] (attention) → [[S4]] (SSM, HiPPO, Cauchy kernel) → [[Mamba]] (selectivity) → [[Transformers Are SSMs]] (semiseparable matrices, SSD) → [[xLSTM]] (exponential gating + matrix memory)

**If you care about alignment:** [[RLHF]] → [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] → [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]]

**If you care about fine-tuning:** [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] → [[RLHF]] → [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]]

**If you care about silicon:** [[Hardware Acceleration for Neural Networks]] → [[Flash Attention]] → [[FlashAttention-2]] → [[KV Cache Optimization]] → [[Hardware-Aware Scan]]

**If you want to understand the linear-RNN family:** [[S4]] → [[RWKV]] → [[RetNet]] → [[Griffin]] → [[Mamba]] → [[Transformers Are SSMs]]
