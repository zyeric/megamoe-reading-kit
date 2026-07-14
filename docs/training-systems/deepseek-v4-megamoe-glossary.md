# DeepSeek-V4 MegaMoE Glossary

Status: layer-aware terminology index

Context metadata:

- Topic: term definitions and layer mapping.
- Layer tags: `algorithm`, `distributed-ep`, `runtime-protocol`,
  `scheduling`, `numerics`, `cuda-model`, `hardware`.
- Owns: short definitions, owner-note pointers, and common misreads.
- Does not own: mechanism walkthroughs or evidence confidence.
- Agent entry: read whenever a term is ambiguous across model, runtime, CUDA,
  and hardware layers.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Use this glossary to avoid collapsing concepts from different layers. The
short definitions are intentionally compact; follow the owner note for details.

## Core Terms

| Term | Layer | Short meaning | Owner note | Common misread |
|---|---|---|---|---|
| MoE | algorithm | Mixture-of-Experts layer: route tokens to selected expert FFNs and combine route outputs | `deepseek-v4-moe-megakernel.md` | Not a communication algorithm by itself |
| expert | algorithm / distributed EP | FFN shard selected by routing; in EP it is owned by a rank | `deepseek-v4-moe-megakernel.md` | Not necessarily local to every token |
| top-k | algorithm | Number of selected expert routes per token | `deepseek-v4-moe-megakernel.md` | Not the same as a top-k combine buffer address |
| top-k slot | algorithm / combine | Lane position of a token's selected route | `deepseek-v4-megamoe-combine.md` | Not necessarily sorted expert ID order |
| `topk_idx` | routing state | Expert IDs selected by router, or invalid sentinel | `deepseek-v4-megamoe-dispatch.md` | Not a combine-slot pointer |
| `topk_weights` | routing / numerics | Route weights for selected experts | `deepseek-v4-megamoe-activation.md` | Not applied in final combine in the public FP8/FP4 path |
| EP | distributed EP | Expert parallelism; experts are partitioned across ranks | `deepseek-v4-moe-megakernel.md` | Not equivalent to one specific all-to-all implementation |
| token-owner rank | distributed EP | Rank that originally owns the local token output | `deepseek-v4-megamoe-combine.md` | Not always the rank that computes a route |
| expert-owner rank | distributed EP | Rank that owns and computes a selected expert | `deepseek-v4-megamoe-dispatch.md` | Not always the source token rank |
| source rank | dispatch | Rank from which a route payload is pulled | `deepseek-v4-megamoe-dispatch.md` | Not necessarily the final combine rank for every buffer |
| destination rank | dispatch / combine | Rank targeted by a peer-memory operation; meaning depends on stage | owner stage note | Must be read with stage context |

## Runtime Protocol Terms

| Term | Layer | Short meaning | Owner note | Common misread |
|---|---|---|---|---|
| symmetric buffer | runtime protocol | Same-layout rank-local buffers with peer-addressable views after rendezvous | `deepseek-v4-megamoe-runtime-protocol.md` | Not a remote procedure call system |
| `sym_buffer.map` | runtime protocol / CUDA | Map a local symmetric-buffer pointer to a peer rank's corresponding address | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-symmetric-memory.md` | Not a special high-level collective |
| rendezvous | runtime setup | Process-group step that exchanges enough state to form peer pointer tables | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-symmetric-memory.md` | Not the hot-path per-token communication itself |
| workspace | runtime protocol | Kernel-owned state area: counters, offsets, metadata, barriers | `deepseek-v4-megamoe-runtime-protocol.md` | Not a single tensor with one semantic meaning |
| pool | runtime protocol | Full logical sequence of routed token occurrences for local expert work | `deepseek-v4-megamoe-runtime-protocol.md` | Not physical storage |
| pool token | runtime protocol | One logical routed token occurrence in the expert-token pool | `deepseek-v4-megamoe-runtime-protocol.md` | Not necessarily one original model token |
| pool block | scheduling / runtime | Logical `BLOCK_M` chunk of an expert-token pool | `deepseek-v4-megamoe-scheduling.md` | Not the same as ring block |
| ring slot | runtime protocol | Reusable physical storage slot for a subset of pool tokens | `deepseek-v4-megamoe-runtime-protocol.md` | Not monotonic logical work |
| ring block | runtime protocol | Group of ring slots guarded by producer-consumer generation state | `deepseek-v4-megamoe-runtime-protocol.md` | Not a wave or CUDA block |
| generation | runtime protocol | Reuse epoch for a physical ring slot | `deepseek-v4-megamoe-runtime-protocol.md` | Not a model generation token |
| `l1_full_count` | runtime protocol | Dispatch publishes L1 ring readiness to Linear1 | `deepseek-v4-megamoe-runtime-protocol.md` | Not a global phase barrier |
| `l1_empty_count` | runtime protocol | Linear1 frees L1 ring slots for reuse | `deepseek-v4-megamoe-runtime-protocol.md` | Not evidence that all waves are done |
| `l2_full_count` | runtime protocol | Activation publishes Linear2 input readiness | `deepseek-v4-megamoe-activation.md` | L2 means Linear2 ring here, not L2 cache |
| `l2_empty_count` | runtime protocol | Linear2 frees L2 ring slots for reuse | `deepseek-v4-megamoe-runtime-protocol.md` | Not a cache event |
| `TokenSrcMetadata` | ownership metadata | Source `{rank_idx, token_idx, topk_idx}` stored per pool token | `deepseek-v4-megamoe-dispatch.md` | Not the router output itself |
| `combine_token_buffer` | combine protocol | Per-top-k-slot route outputs on token-owner rank before final sum | `deepseek-v4-megamoe-combine.md` | Not the final output tensor |
| `nvlink_barrier` | runtime protocol | Phase-level cross-rank visibility / ordering point in the fused protocol | `deepseek-v4-megamoe-runtime-protocol.md` | Not proof of cross-node RDMA support |

## Scheduling And Kernel Terms

| Term | Layer | Short meaning | Owner note | Common misread |
|---|---|---|---|---|
| wave | scheduling | High-level work window over a subset of expert blocks | `deepseek-v4-megamoe-scheduling.md` | Not warp, ring block, or GEMM tile |
| persistent kernel | kernel pattern | Long-lived kernel with resident workers that traverse work dynamically | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md` | Not a guarantee that no thread ever waits |
| persistent worker | scheduling | Resident CTA / warp role polling and executing assigned work | `deepseek-v4-megamoe-scheduling.md` | Not a CPU thread |
| spin-wait | kernel pattern | Device-side polling on counters / flags | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md` | Not host-side sleep or heartbeat |
| `BLOCK_M` | GEMM shape | Number of token rows in one M block | `deepseek-v4-megamoe-gemm.md` | Not the full expert token count |
| `BLOCK_N` | GEMM shape | Number of output-channel columns in one N block | `deepseek-v4-megamoe-gemm.md` | Not a rank count |
| `BLOCK_K` | GEMM shape | Reduction-dimension tile size | `deepseek-v4-megamoe-gemm.md` | Not top-k |
| GEMM tile | compute tiling | One M x N output tile, reduced over K tiles | `deepseek-v4-megamoe-gemm.md` | Not the same as communication chunk |
| K-tile stage | GEMM pipeline | Shared-memory pipeline stage over the reduction dimension | `deepseek-v4-megamoe-gemm.md` | Not MoE pipeline stage |
| CTA | CUDA programming model | Cooperative Thread Array, the CUDA block abstraction | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md` | Not one physical SM |
| warp | CUDA programming model | Group of 32 CUDA threads scheduled together | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md` | Not an SM or a wave |
| warpgroup | CUDA programming model | Group of warps cooperating for tensor-core operations | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md` | Not a model expert group |
| SM | hardware | Streaming Multiprocessor executing CTAs / warps | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md` | Not determined by `sm90` / `sm100` number |
| SM90 / SM100 | ISA / architecture | NVIDIA architecture generation targets for Hopper / Blackwell | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md` | Not the number of SMs |

## Numerics And Hardware Terms

| Term | Layer | Short meaning | Owner note | Common misread |
|---|---|---|---|---|
| BF16 | numerics | 16-bit floating type used for model hidden states and final outputs | `deepseek-v4-megamoe-quantization.md` | Not the internal GEMM accumulator type |
| FP8 E4M3 | numerics | 8-bit float payload format for activations in the public path | `deepseek-v4-megamoe-quantization.md` | Not a standalone value without scale |
| FP4 E2M1 | numerics | 4-bit float payload format for weights in the public path | `deepseek-v4-megamoe-quantization.md` | Not the V3 FP8 training recipe |
| UE8M0 | numerics | Unsigned 8-bit exponent-only scale-factor representation | `deepseek-v4-megamoe-quantization.md` | Not an FP8 payload format |
| SFA | numerics / GEMM | Activation scale-factor metadata consumed by block-scaled UMMA | `deepseek-v4-megamoe-gemm.md` | Not loaded exactly like large payload tiles |
| SFB | numerics / GEMM | Weight scale-factor metadata consumed by block-scaled UMMA | `deepseek-v4-megamoe-gemm.md` | Not the weight payload |
| UTCCP | Blackwell / GEMM | Instruction path used to move scale-factor metadata into TMEM for UMMA | `deepseek-v4-megamoe-gemm.md` | Not the general payload load path |
| TMA | CUDA / hardware | Tensor Memory Accelerator for asynchronous tensor loads / stores | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md` | Not TMEM |
| TMEM | Blackwell / hardware | Tensor memory used by UMMA accumulators and epilogues | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md` | Not global memory and not TMA |
| MMA / UMMA | tensor core | Matrix multiply-accumulate instruction family; UMMA is Blackwell-style | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md` | Not CUDA-core scalar math |
| shared memory | GPU memory | Explicit CTA-local scratchpad, often sharing on-chip capacity with L1 | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md` | Not a global cache |
| L2 cache | GPU memory | GPU-wide cache and coherence/performance layer | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md` | Not Linear2 when written as `l2_acts` |
| HBM | GPU memory | High-bandwidth device memory for large tensors | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md` | Not on-chip scratchpad |
| NVLink / NVSwitch | interconnect | Intra-node / rack GPU peer transport fabric | `https://zyeric.github.io/gpu-hardware-notes/notes/nvidia-gpu-generations.md` | Not equivalent to cross-node RDMA |

## Naming Collision Warnings

- `L1` can mean Linear1 ring / stage in these notes, not L1 cache.
- `L2` can mean Linear2 ring / stage in these notes, not L2 cache.
- `wave` is a MegaMoE scheduling window, not CUDA warp.
- `shared_expert_count` in public code should not be read as model-level
  "shared expert" architecture.
- `generation` in ring buffers is a reuse epoch, not text generation.
