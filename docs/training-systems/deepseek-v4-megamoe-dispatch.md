# DeepSeek-V4 / DeepGEMM MegaMoE Dispatch Notes

Status: reorganized dispatch-only note

Context metadata:

- Topic: lowering already-computed route state into local Linear1 inputs.
- Layer tags: `stage-dispatch`, `distributed-ep`, `runtime-protocol`.
- Owns: metadata push, deterministic source-rank order, L1 token/SF pull,
  `TokenSrcMetadata` production, and `l1_full_count` publication.
- Does not own: generic ring-buffer definitions, wave scheduling, GEMM body,
  activation, or final combine.
- Agent entry: read after runtime protocol when the task asks "how does a
  routed token become a local Linear1 input?".

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - symmetric buffer, pool/ring/counter protocol, capacity, and resource lanes.
- [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md) -
  waves, persistent workers, and scheduling windows.
- [`deepseek-v4-megamoe-quantization.md`](deepseek-v4-megamoe-quantization.md) -
  `x` / `x_sf`, route-weight dtype, and scale-factor payloads.
- [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) - how ready L1
  ring slots become Linear1 GEMM inputs.
- [`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-cuda-symmetric-memory)
  - peer-addressability background.

Scope:

- Explain the dispatch lowering in the public DeepGEMM SM100 FP8/FP4 MegaMoE
  forward path.
- Focus on how top-k route state is converted into local L1 ring-buffer inputs
  and source-token metadata.
- Do not re-explain the full symmetric-memory protocol, scheduling waves, GEMM
  K-tile pipeline, or combine reduction here.

## What Moved Out Of This File

The first pass used dispatch as the first place to collect many concepts. Those
concepts now live in more specific notes:

| Old topic in dispatch note | New owner |
|---|---|
| symmetric-memory setup and transport caveats | [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md), [`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-cuda-symmetric-memory) |
| tensor / buffer glossary | [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md) |
| pool, ring slots, generation counters | [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md) |
| wave scheduling examples | [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md) |
| resource partition and GEMM stage counts | [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md), [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) |

This file should now answer one question:

```text
Given already-computed routing/top-k state,
how does the fused kernel prepare local Linear1 inputs?
```

## One-Line Model

Dispatch is a device-side expert-parallel lowering:

```text
remote token-owner symmetric buffers
  -> source-rank counts and deterministic source order
  -> local expert-token pool
  -> L1 ring slots: l1_acts, l1_acts_sf, route weight
  -> TokenSrcMetadata for later combine write-back
  -> l1_full_count publishes readiness to Linear1
```

The output is not `y`; it is a set of local Linear1 inputs plus enough metadata
to route Linear2 results back to original token owners.

## Code Anchors

DeepGEMM public files previously inspected:

- `deep_gemm/mega/__init__.py` - Python `SymmBuffer` setup, input views, and
  public `fp8_fp4_mega_moe` / `bf16_mega_moe` wrappers.
- `csrc/apis/mega.hpp` - symmetric-buffer size calculation and tensor view
  shapes.
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` - workspace counters,
  metadata arrays, ring-buffer counters, and `TokenSrcMetadata`.
- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` - wave-local Linear1 /
  Linear2 scheduler state machine.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` - current
  public SM100 FP8/FP4 fused kernel implementation.
- `csrc/jit_kernels/heuristics/mega_moe.hpp` - JIT policy for block shapes,
  warp/thread layout, wave size, pull chunk size, and GEMM pipeline stages.
- `tests/test_mega_moe.py` - executable contract, default shapes, and benchmark
  accounting.

## Dispatch Position In The Lowering

Distributed EP reference:

```text
for each local token:
  route to top-k experts
  send each route payload to the rank that owns the expert

for each expert-owner rank:
  run local experts on received routes

send route outputs back to original token-owner ranks
sum top-k route outputs
```

MegaMoE dispatch rewrite:

```text
before fused kernel:
  routing/top-k already computed
  x, x_sf, topk_idx, topk_weights copied into symmetric-buffer views

inside fused kernel:
  expert-owner rank pulls remote route payloads into local L1 ring slots
  source identity is recorded in TokenSrcMetadata
  Linear1 consumes local L1 ring slots when l1_full_count is published
```

So "dispatch" in this note means:

```text
prepare local Linear1 inputs for routes owned by this rank's local experts
```

not:

```text
run router / top-k
perform a standalone all-to-all kernel
compute expert FFN output
combine final token output
```

## Inputs And Outputs

Dispatch inputs:

| Input | Meaning |
|---|---|
| `x` | source-token FP8 payload |
| `x_sf` | packed source-token scale factors |
| `topk_idx` | selected expert IDs, or `-1` for invalid / masked routes |
| `topk_weights` | route weights applied later in the Linear1 activation epilogue |
| expert-count metadata | per-source-rank counts for each local expert |
| symmetric peer pointers | addressability for remote rank buffers |

Dispatch outputs:

| Output | Meaning | Consumer |
|---|---|---|
| `l1_acts` | local Linear1 input payload | Linear1 GEMM |
| `l1_acts_sf` | local Linear1 input scale factors | Linear1 GEMM |
| local route weight buffer | one route weight per local pool token | activation epilogue |
| `TokenSrcMetadata` | original rank, token index, and top-k slot | Linear2 write-back |
| `l1_full_count` | readiness publication for ring block generation | Linear1 scheduler |

## Routing Is Outside MegaMoE

The routing / top-k kernel is outside the public MegaMoE fused kernel. Its
outputs are ordinary CUDA tensors that are copied into the symmetric buffer
before the fused call:

```text
router/top-k kernel
  -> topk_idx, topk_weights
  -> symmetric-buffer input views
  -> MegaMoE fused kernel starts
```

Therefore, public DeepGEMM MegaMoE does not pipeline router/top-k computation
with dispatch pull. Dispatch starts after `x`, `x_sf`, `topk_idx`, and
`topk_weights` are already available in symmetric-buffer views.

## Metadata Push

Before pulling the full payload, ranks need enough metadata to know how many
routes each local expert will receive from each source rank.

Conceptually:

```text
source rank:
  inspect local topk_idx
  count routes targeting remote rank's local experts
  publish per-destination metadata / counts through symmetric memory

destination rank:
  use per-source-rank counts
  compute local pool offsets
  plan which source tokens to pull for each local expert
```

The important output is not just a total count; dispatch needs enough
deterministic structure to map a logical pool position back to:

```text
source rank
source token index
source top-k slot
destination local expert
```

This is the information later stored as `TokenSrcMetadata`.

## `TokenSrcMetadata`

During dispatch pull, the expert-owner rank records how each local pool token
maps back to the original token-owner rank and top-k slot:

```cpp
*workspace.get_token_src_metadata_ptr(pool_token_idx) =
    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
```

The metadata type is:

```cpp
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};
```

This metadata is stored for the full logical expert-token pool, not only in a
reusable ring slot:

```text
pool token -> original rank, original token, original top-k slot
```

Reason: by the time Linear2 writes route outputs back, the temporary physical
L1/L2 ring slot may already have been reused. Linear2 write-back needs logical
source identity, not the temporary storage slot.

## Pull Into L1 Ring Buffer

After metadata/count planning, dispatch pulls payloads into local L1 ring slots:

```text
remote symmetric source x / x_sf
  -> destination rank TMA / peer load path
  -> local l1_acts / l1_acts_sf ring slot
```

The payload being pulled is already in the source rank's symmetric global
memory before the MegaMoE kernel starts. Symmetric memory does not first stage
it into a separate RPC buffer; the destination rank maps a peer address and
loads from that address into local storage.

Dispatch also copies route weight state so that the Linear1 activation epilogue
can apply `topk_weights` before Linear2:

```text
topk_weights[src_token, src_topk_slot]
  -> local per-pool-token route weight
  -> activation epilogue multiplies after SwiGLU
```

## Source-Rank Interleaving Order

Within one local expert, per-source-rank counts define how many routes come
from each source rank. The implementation maps logical pool slots to source
ranks in a deterministic interleaved order rather than draining one source rank
completely before moving to the next.

Simplified example:

```text
local expert E0 receives:
  rank0: 3 routes
  rank1: 1 route
  rank2: 2 routes

one deterministic interleaved pool order can look like:
  r0_0, r1_0, r2_0, r0_1, r2_1, r0_2
```

Why this matters:

- it avoids a pool layout that is dominated by one source rank at a time;
- it makes pull work more evenly distributed across source ranks;
- it gives a deterministic logical order for metadata and later write-back.

The exact order should be read from the code path, but the useful mental model
is:

```text
per-rank counts
  -> deterministic source-rank interleaving
  -> logical pool token order
  -> physical ring slot via pool_token_idx % num_ring_tokens
```

This is still dispatch-specific because it decides which remote source payload
is pulled for each local pool position.

## Readiness Publication

After dispatch writes a ring block's L1 payload and metadata, it publishes
readiness through `l1_full_count`:

```text
dispatch producer:
  write l1_acts / l1_acts_sf / route weight / TokenSrcMetadata
  publish l1_full_count[ring_block]

Linear1 consumer:
  wait until expected l1_full_count generation is visible
  run Linear1 GEMM on that ring block
  later publish l1_empty_count so the slot can be reused
```

The full counter/generation model lives in
[`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md).

## Spin-Wait In Dispatch Context

Dispatch may spin while waiting for a reusable L1 ring slot to become empty or
for cross-rank metadata to become visible. Here "spin-wait" means resident GPU
threads repeatedly check a device-visible condition:

```text
while counter_or_flag != expected:
  keep polling
```

This is not a host-side polling loop. It is a device-side synchronization
pattern inside the persistent kernel.

For the broader explanation of which counters protect which ring slots, use
the runtime protocol note.

## Worked EP8 Sketch

Suppose EP8: eight ranks participate, and each rank owns a subset of experts.
For one destination rank:

1. Every source rank has local tokens and `topk_idx`.
2. Source ranks publish counts for routes targeting this destination rank's
   local experts.
3. The destination rank computes local expert pool offsets.
4. For local expert `E`, the destination rank interleaves source ranks according
   to their per-rank counts.
5. For each logical pool token, it pulls `x`, `x_sf`, and route weight into a
   local L1 ring slot.
6. It records `TokenSrcMetadata(pool_token)` so Linear2 can push the route
   output back to the original token owner.
7. It publishes `l1_full_count` for the ring block.

From this point, the route is no longer a remote token for Linear1. It is a
local pool token with local L1 input storage and source metadata.

## Boundary To Linear1

Dispatch ends when the following are true for a ring block generation:

```text
l1_acts and l1_acts_sf are valid
route weight is available
TokenSrcMetadata is recorded
l1_full_count is published
```

Linear1 begins when its scheduler observes the relevant `l1_full_count` and
selects a GEMM tile for that ready work.

Details after this boundary belong to:

- [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) for TMA / UMMA /
  TMEM GEMM mechanics;
- [`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md) for
  top-k weight application and L2 ring publish.

## Reading Checklist

When reading dispatch code, identify:

1. Which route metadata is already available before the fused kernel starts?
2. Which remote rank owns the source token?
3. Which local expert owns the selected route?
4. What logical `pool_token_idx` is assigned?
5. Which physical L1 ring slot does that pool token use?
6. Which source metadata is stored for later combine write-back?
7. Which counter publishes readiness to Linear1?

If a detail is about "what is a ring buffer" or "how symmetric memory works",
use the runtime / hardware notes instead of expanding this dispatch note again.

## Open Follow-Ups

- Pin the exact source code block that implements the deterministic source-rank
  interleaving order.
- Verify whether production serving code changes capacity selection or fallback
  behavior for pathological imbalance.
- Compare this dispatch pull path against a DeepEP-style standalone all-to-all
  baseline once a matching benchmark is available.
