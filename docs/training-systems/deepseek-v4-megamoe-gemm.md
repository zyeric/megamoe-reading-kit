# DeepSeek-V4 / DeepGEMM MegaMoE GEMM Pipeline Notes

Status: first-pass Linear1 / Linear2 shared GEMM notes

Context metadata:

- Topic: shared Linear1 / Linear2 TMA + UMMA + TMEM GEMM machinery.
- Layer tags: `stage-gemm`, `numerics`, `cuda-model`, `hardware`.
- Owns: logical GEMM semantics, tensor maps, shared-memory K-tile pipeline,
  scale-factor flow into UMMA, TMEM accumulation, and GEMM determinism edges.
- Does not own: dispatch route ownership, activation epilogue math, or final
  combine write-back.
- Agent entry: read when the task mentions TMA, UMMA, TMEM accumulators,
  UTCCP, GEMM tile order, or Linear1/Linear2 common structure.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - L1/L2 readiness protocol and ring-slot boundaries.
- [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md) -
  how `l1_acts` / `l1_acts_sf` ring slots become ready for Linear1.
- [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md) -
  waves, pool blocks, ring blocks, and scheduler traversal.
- [`deepseek-v4-megamoe-quantization.md`](deepseek-v4-megamoe-quantization.md)
  - FP8/FP4 payloads, SFA/SFB layouts, and block-scaled matmul semantics.
- [`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md) -
  Linear1 epilogue, SwiGLU, `l2_acts`, and TMEM accumulator handoff.
- [`gpu-hardware-notes/notes/cuda-kernel-patterns.md`](https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md)
  - TMA, UMMA, TMEM, epilogue, and producer-consumer pipeline patterns.

Scope:

- Track the public DeepGEMM SM100 FP8/FP4 MegaMoE forward GEMM body.
- Treat Linear1 and Linear2 together because they share the same TMA / UMMA /
  TMEM machinery.
- Keep Linear1 activation epilogue in the activation note.
- Keep Linear2 remote write-back and final top-k-slot sum for the combine note.

Current evidence level:

- Code-checked against public DeepGEMM `main` files:
  `tests/test_mega_moe.py`,
  `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`,
  `csrc/jit_kernels/heuristics/mega_moe.hpp`,
  `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`, and
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`.
- This note describes the public SM100 FP8/FP4 forward path. Do not generalize
  it to unpublished backward kernels or non-SM100 implementations without new
  source evidence.

## One-Line Model

Linear1 and Linear2 share one persistent tiled GEMM engine:

```text
scheduler returns (phase, expert, m_block, n_block)
  -> TMA load A/SFA tile into shared memory
  -> TMA load B/SFB tile into shared memory
  -> UTCCP copies SFA/SFB scale metadata to TMEM
  -> block-scaled UMMA accumulates FP32 C tile in TMEM
  -> epilogue consumes the complete C tile
```

The two phases differ mainly by tensor maps and epilogues:

| Phase | Logical A | Logical B | Accumulator consumer |
|---|---|---|---|
| Linear1 | `l1_acts` FP8 + `l1_acts_sf` | `l1_weights` FP4 + `l1_weights_sf` | SwiGLU / top-k / FP8 `l2_acts` epilogue |
| Linear2 | `l2_acts` FP8 + `l2_acts_sf` | `l2_weights` FP4 + `l2_weights_sf` | BF16 route-output write-back epilogue |

So it is useful to read them together until the epilogue branch:

```text
same scheduler
same TMA A/B/SF stage pipeline
same block-scaled UMMA instruction family
same FP32 TMEM accumulator handoff
different output interpretation
```

## Code Anchors

DeepGEMM paths:

- `tests/test_mega_moe.py` - creates BF16 reference tensors, quantizes inputs /
  weights for the FP8/FP4 path, and compares fused output to the legacy
  baseline.
- `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` - builds TMA tensor maps
  for L1/L2 activations, weights, scale factors, and L1 output.
- `csrc/jit_kernels/heuristics/mega_moe.hpp` - chooses `BLOCK_M`, `BLOCK_N`,
  `BLOCK_K`, `num_stages`, thread layout, and wave size.
- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` - returns
  `BlockPhase::Linear1` or `BlockPhase::Linear2` plus expert / M / N block
  indices.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` - resident
  kernel, TMA load warps, UMMA issue warp, TMEM barriers, and epilogue warps.

Important variables:

| Name | Meaning |
|---|---|
| `BLOCK_M` | token rows in one logical M block |
| `BLOCK_N` | output-channel columns in one N block; public FP8/FP4 path asserts `128` |
| `BLOCK_K` | reduction tile width loaded per GEMM pipeline stage |
| `LOAD_BLOCK_M = BLOCK_M / 2` | A tile rows per CTA in the 2-CTA cluster path |
| `LOAD_BLOCK_N = BLOCK_N` | B tile columns per load |
| `kNumStages` / `num_stages` | shared-memory stages for the TMA/UMMA K-tile pipeline |
| `stage_idx`, `phase` | rotating stage index and barrier phase for shared-memory pipeline reuse |
| `kNumEpilogueStages = 2` | separate TMEM accumulator stages between UMMA and epilogue |
| `full_barriers`, `empty_barriers` | shared-memory stage producer-consumer barriers |
| `tmem_full_barriers`, `tmem_empty_barriers` | UMMA accumulator producer-consumer barriers |

## Logical GEMM Semantics

For each local expert and each routed-token block, both Linear1 and Linear2
are logically:

```text
C[M, N] = A[M, K] @ B[N, K]^T
```

For the public FP8/FP4 path:

```text
A: FP8 E4M3 activation payload + SFA UE8M0 scale factors
B: FP4 E2M1 weight payload + SFB UE8M0 scale factors
C: FP32 accumulator in TMEM
```

The phase determines shapes:

| Phase | M | N | K |
|---|---:|---:|---:|
| Linear1 | routed tokens for one expert block | `2 * intermediate_hidden` | `hidden` |
| Linear2 | same routed-token block after activation | `hidden` | `intermediate_hidden` |

The code names these compile-time shapes:

```cpp
L1_SHAPE_N = kIntermediateHidden * 2;
L1_SHAPE_K = kHidden;
L2_SHAPE_N = kHidden;
L2_SHAPE_K = kIntermediateHidden;
```

DeepGEMM's SM100 implementation comments say it "always swap[s] A/B" for the
UMMA instruction. This is an implementation detail of the instruction dataflow:
the logical GEMM is still activation rows times weight rows, but the UMMA
descriptor call passes `b_desc` then `a_desc` and writes the accumulator tile
in the layout the epilogue expects.

## Tensor Maps: What Changes Between Linear1 And Linear2

The runtime builds separate TMA descriptors:

```text
Linear1 A:
  tensor_map_l1_acts
  tensor_map_l1_acts_sf

Linear1 B:
  tensor_map_l1_weights
  tensor_map_l1_weights_sf

Linear2 A:
  tensor_map_l2_acts
  tensor_map_l2_acts_sf

Linear2 B:
  tensor_map_l2_weights
  tensor_map_l2_weights_sf
```

Inside the kernel, the TMA load warps choose descriptors by `block_phase`:

```cpp
tensor_map_a_ptr =
    block_phase == Linear2 ? &tensor_map_l2_acts : &tensor_map_l1_acts;
tensor_map_sfa_ptr =
    block_phase == Linear2 ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

tensor_map_b_ptr =
    block_phase == Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
tensor_map_sfb_ptr =
    block_phase == Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;
```

Readiness also differs by phase:

```text
Linear1:
  wait l1_full_count[ring_block]
  dispatch has pulled full token block into l1_acts

Linear2:
  wait l2_full_count[ring_block]
  Linear1 epilogue has produced all l2_acts chunks for this ring generation
```

This is the point where the shared GEMM engine connects to the larger fused
MoE pipeline.

## Shared-Memory Stage Pipeline

The GEMM body has a K-tile pipeline:

```text
stage 0:
  TMA load A/B/SF tile for K block 0 into shared memory

stage 1:
  UMMA consumes K block 0
  TMA load A/B/SF tile for K block 1

stage 2:
  UMMA consumes K block 1
  TMA load A/B/SF tile for K block 2
```

This is controlled by:

```cpp
uint32_t stage_idx = 0, phase = 0;

auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
};
```

Do not confuse this with MoE waves or TMEM epilogue stages:

| Mechanism | Unit | Storage | Purpose |
|---|---|---|---|
| MoE wave | group of local experts / blocks | scheduler state + ring buffers | expose enough expert blocks while bounding workspace |
| GEMM `kNumStages` | K-tile stage | shared memory `smem_a/smem_b/smem_sfa/smem_sfb` | overlap TMA tile load with UMMA compute |
| Epilogue `kNumEpilogueStages` | C-tile accumulator stage | TMEM | overlap UMMA producer with epilogue consumer |

The TMA load warps wait until a shared-memory stage is released:

```cpp
shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
```

Then A/SFA and B/SFB are copied into shared memory and the stage's full barrier
is updated:

```cpp
tma::copy(..., shared_storage.smem_a[stage_idx], ...);
tma::copy(..., shared_storage.smem_sfa[stage_idx], ...);

tma::copy(..., shared_storage.smem_b[stage_idx], ...);
tma::copy(..., shared_storage.smem_sfb[stage_idx], ...);
```

The UMMA issue warp waits for both A-side and B-side loads:

```cpp
shared_storage.full_barriers[stage_idx].wait(phase);
```

After UMMA has consumed the stage, it arrives on the `empty_barriers` object so
the TMA load warps can reuse that shared-memory stage for a later K tile.

## Scale-Factor Flow Into UMMA

The FP8/FP4 path is block-scaled. Payload tiles and scale-factor tiles move
through different shared-memory arrays:

```text
A payload  -> smem_a
A scales   -> smem_sfa
B payload  -> smem_b
B scales   -> smem_sfb
```

The global-to-shared movement for scale factors is still TMA. There are two
movement steps:

```text
global memory -> shared memory:
  TMA reads payload tensors and scale-factor tensors

shared memory -> TMEM:
  UTCCP copies only scale-factor tiles into the TMEM scale-factor area
```

For the A side, the TMA load warp issues both payload and SFA copies:

```cpp
tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
    tensor_map_a_ptr, ..., shared_storage.smem_a[stage_idx], k_idx, ring_m_idx, 2);

tma::copy<SF_BLOCK_M, 1, 0>(
    tensor_map_sfa_ptr, ..., shared_storage.smem_sfa[stage_idx],
    sfa_ring_m_idx, sfa_k_idx, 2);
```

For the B side, it does the same for weights and SFB:

```cpp
tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
    tensor_map_b_ptr, ..., shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);

tma::copy<BLOCK_N, 1, 0>(
    tensor_map_sfb_ptr, ..., shared_storage.smem_sfb[stage_idx],
    sfb_n_idx, sfb_k_idx, 2);
```

So the scale-factor path is not "no TMA". It is:

```text
l1/l2 activation SF or weight SF in global memory
  -> TMA
  -> smem_sfa / smem_sfb
  -> UTCCP
  -> TMEM scale-factor columns
```

Before issuing UMMA for a `UMMA_BLOCK_K = 128` chunk, the kernel performs the
second step: it copies SFA and SFB from shared memory into TMEM using SM100
UTCCP instructions:

```cpp
using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;

cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + ...);
cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + ...);
```

`UTCCP` is treated here as a code-level name, not a fully decoded public
acronym. The source evidence tells us what it does in this kernel:

```text
SM100_UTCCP_4x32dp128bit_2cta
  reads scale-factor tiles from shared memory using sf_desc
  writes them to TMEM columns starting at kTmemStartColOfSFA/SFB
  expects the 4x32 / 128-element scale-factor layout used by DeepGEMM
  matches the 2-CTA SM100 UMMA dataflow
```

The "metadata" in this phrase is scale-factor metadata:

```text
one UE8M0 scale byte per 32 K values
four scale bytes packed into one uint32_t in the tensor view
```

It is not routing metadata such as expert IDs, source ranks, or token indices.

Then the block-scaled UMMA runtime descriptor references those scale-factor
locations:

```cpp
runtime_instr_desc =
    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);

ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
    b_desc, a_desc,
    accum_stage_idx * UMMA_N,
    is_accumulate,
    runtime_instr_desc,
    kTmemStartColOfSFB,
    kTmemStartColOfSFA);
```

This is the hardware-level counterpart of the quantized matmul formula in the
quantization note:

```text
sum over K groups:
  A_sf[m, g] * B_sf[n, g] * dot(A_q[m, group], B_q[n, group])
```

The reason the non-scale payload does not use UTCCP is the SM100 block-scaled
UMMA operand contract. The instruction is effectively shaped like:

```text
umma_block_scaled(
  A_shared_descriptor,
  B_shared_descriptor,
  C_tmem_address,
  SFA_tmem_columns,
  SFB_tmem_columns)
```

Payload operands are large matrix tiles and are consumed directly from shared
memory through `a_desc` / `b_desc`. Scale factors are small, per-32-K metadata
with a different access pattern, and this instruction path expects them in
TMEM scale-factor columns. Therefore:

```text
A/B payload:
  global -> TMA -> shared memory
  UMMA reads through a_desc / b_desc

SFA/SFB:
  global -> TMA -> shared memory
  UTCCP -> TMEM
  UMMA reads through scale-factor column arguments

C accumulator:
  UMMA writes TMEM
```

This is best understood as an ISA / hardware dataflow constraint, not a MoE
semantic choice. The software layout rewrites and UTCCP copy exist to satisfy
the operand placement required by block-scaled UMMA.

## UMMA Accumulation Order And TMEM Handoff

For one scheduled block, UMMA loops in a fixed nested order:

```text
for k_block_idx in K / BLOCK_K:
  for umma_k_block_idx in BLOCK_K / UMMA_BLOCK_K:
    for k in UMMA_BLOCK_K / UMMA_K:
      issue UMMA
```

The first UMMA for a C tile clears / initializes the accumulator; later UMMAs
accumulate into the same TMEM C tile:

```cpp
is_accumulate = k_block_idx > 0 or umma_k_block_idx > 0 or k > 0;
```

The epilogue does not consume partial K reductions. `tmem_full` is published
only after the last K block for that logical C tile:

```cpp
empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
```

This gives the epilogue a complete FP32 C tile:

```text
all K tiles accumulated
  -> tmem_full_barrier
  -> Linear1 or Linear2 epilogue consumes complete C tile
```

The shared-memory K-tile pipeline and the TMEM accumulator pipeline are
therefore nested but distinct:

```text
TMA/UMMA K pipeline:
  global -> shared -> Tensor Core -> TMEM accumulator

TMEM epilogue pipeline:
  complete TMEM C tile -> epilogue -> output buffer
```

## Scheduler Order And Determinism

The scheduler returns a deterministic sequence of logical blocks for each
resident worker:

```text
(BlockPhase, local_expert_idx, m_block_idx, n_block_idx)
```

At a high level:

```text
for each wave:
  expose Linear1 blocks for experts in the wave
  then expose Linear2 blocks for those experts
  then move to the next wave
```

The source state machine does this by starting with `next_phase = Linear1`,
switching to `Linear2` after L1 blocks in the current wave are assigned, and
then returning to `Linear1` for the next wave.

Within one C tile, deterministic properties in the public forward GEMM body:

- the K-block traversal order is fixed;
- the UMMA sub-K traversal order is fixed;
- scale-factor locations are determined by tensor maps and K block indices;
- the epilogue consumes only complete C tiles;
- there is no split-K atomic reduction across multiple CTAs for the same C tile
  in this path.

This is a forward-path statement. It does not automatically answer backward
determinism, because backward may introduce `dW` / `dX` reductions, atomics,
split-K, or cross-token accumulation patterns that are not present in this
public forward kernel.

## Boundary To Activation And Combine

The GEMM body ends at a complete FP32 accumulator tile in TMEM.

Linear1 handoff:

```text
Linear1 UMMA
  -> FP32 TMEM C tile
  -> activation epilogue
  -> FP8 l2_acts + l2_acts_sf
  -> l2_full_count
```

That path is covered by
[`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md).

Linear2 handoff:

```text
Linear2 UMMA
  -> FP32 TMEM C tile
  -> BF16 route output
  -> remote combine_token_buffer slot
  -> final top-k slot sum
```

That path should be covered by a separate combine note, because the special
logic is no longer GEMM: it uses source-token metadata, peer writes, a
pre-combine barrier, and final per-token top-k slot summation.

## Open Questions

- Need generated PTX / SASS or NVIDIA docs to fully decode the low-level UMMA /
  UTCCP instruction shapes and TMEM column addressing.
- Need GPU profiling to decide when the GEMM body is Tensor-Core-bound versus
  waiting on TMA, shared-memory stages, scale-factor movement, or epilogue.
- Need a backward-path source before making any training-gradient determinism
  claims.
