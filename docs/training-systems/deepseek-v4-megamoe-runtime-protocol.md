# DeepSeek-V4 / DeepGEMM MegaMoE Runtime Protocol Notes

Status: reorganized cross-stage protocol note

Context metadata:

- Topic: cross-stage runtime objects used by the fused MegaMoE forward path.
- Layer tags: `runtime-protocol`, `distributed-ep`, `scheduling`.
- Owns: symmetric-buffer role, pool/ring concepts, generation counters,
  source metadata lifetime, phase barriers, and resource-lane model.
- Does not own: source-rank interleaving order, GEMM K-tile pipeline, or
  exact hardware instruction semantics.
- Agent entry: read this before dispatch, scheduling, activation, or combine
  whenever a question mentions pool, ring, counters, or symmetric buffers.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md) -
  dispatch-specific metadata push and L1 ring-buffer pull.
- [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md) -
  waves, persistent workers, work windows, and scheduling bubbles.
- [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) - Linear1 /
  Linear2 tiled GEMM body.
- [`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md) -
  Linear1 epilogue and L2 ring-buffer publish.
- [`deepseek-v4-megamoe-combine.md`](deepseek-v4-megamoe-combine.md) -
  Linear2 route write-back and final local combine.
- [`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-cuda-symmetric-memory)
  - symmetric-memory addressability and transport caveats.

Scope:

- Keep cross-stage runtime objects in one place: symmetric buffer, pool,
  ring slots, counters, source-token metadata, and resource lanes.
- Keep dispatch mechanics out of this note unless they define a reusable
  protocol object.
- Keep CUDA / GPU hardware background in the hardware notes.

## One-Line Model

MegaMoE forward is easiest to read as a device-side protocol over shared
runtime state:

```text
symmetric peer-visible buffers
  + full logical expert-token pool
  + reusable L1/L2 ring slots
  + per-slot generation counters
  + source-token metadata
  + phase barriers
```

The protocol lets dispatch, Linear1, activation, Linear2, and combine run
inside one persistent kernel without materializing a clean sequence of
standalone all-to-all / grouped-GEMM / combine kernels.

## Why This Note Exists

The first pass put many protocol concepts in the dispatch note because dispatch
was the first concrete code path we read. That made later reading harder:

```text
dispatch-specific:
  metadata push, source-rank order, pulling x/x_sf/topk_weight into L1

cross-stage protocol:
  symmetric buffer, pool, ring slots, counters, capacity, phase barriers

scheduling policy:
  wave windows, persistent workers, block/tile traversal, resource split
```

This note separates the second category from the first.

## Runtime Object Map

| Object | Layer | Meaning | Main Consumer |
|---|---|---|---|
| `SymmBuffer` / `sym_buffer` | runtime addressability | Peer-visible same-layout buffers, mapped by rank and offset | dispatch pull, metadata writes, combine write-back |
| `workspace` | kernel protocol | Counters, metadata arrays, offsets, barriers | all fused stages |
| full expert-token pool | logical execution | Per-local-expert logical token sequence after routing | dispatch, GEMM scheduler, combine write-back |
| L1 ring slots | buffer protocol | Reusable slots holding Linear1 input payload / SF / top-k weight | Linear1 |
| L2 ring slots | buffer protocol | Reusable slots holding Linear2 input activation / SF | Linear2 |
| `TokenSrcMetadata` | ownership metadata | `{rank_idx, token_idx, topk_idx}` for each logical pool token | Linear2 epilogue / combine write-back |
| `l1_full_count` / `l1_empty_count` | producer-consumer counter | Dispatch publishes L1 readiness; Linear1 frees physical slots | dispatch / Linear1 |
| `l2_full_count` / `l2_empty_count` | producer-consumer counter | Activation publishes L2 readiness; Linear2 frees physical slots | activation / Linear2 |
| `combine_token_buffer` | route-output buffer | Per-top-k-slot BF16 route outputs on token-owner rank | final combine |
| `nvlink_barrier` | phase barrier | Cross-rank visibility / ordering point | metadata handoff, pre-combine reduction |

## Symmetric-Memory Setup

The public DeepGEMM MegaMoE wrapper uses PyTorch symmetric memory to allocate a
same-layout buffer per rank and perform rendezvous before the fused kernel.
The fused kernel then receives enough information to map an address for a
remote rank:

```text
rank-local symmetric allocation
  -> same shape / layout on all ranks
  -> rendezvous / peer pointer exchange
  -> kernel-side sym_buffer.map(local_layout_ptr, remote_rank)
```

Use the hardware note for the lower-level addressability model:
[`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-cuda-symmetric-memory).

Important caveat:

```text
symmetric memory provides peer addressability
it does not by itself provide the MegaMoE protocol
```

Correctness still depends on metadata layout, counters, acquire/release
ordering, barriers, and deterministic ownership rules in the fused kernel.

## Transport Caveat

For the current public DeepGEMM path, the evidence points to an NVLink-domain
peer-memory design:

- the kernel uses `comm::nvlink_barrier`;
- public comments refer to NVLink remote writes;
- benchmark accounting separates "NVLink bytes" for dispatch pull and combine
  write-back;
- the kernel directly maps peer buffer addresses from device code.

Do not generalize this note to cross-node RDMA / IB training communication
without separate source evidence.

## Tensor And Buffer Semantics

The public FP8/FP4 MegaMoE path has a few tensor names that are easy to mix:

| Name | Meaning | Location / Lifetime |
|---|---|---|
| `x` | quantized hidden-state payload for source tokens | symmetric global buffer before kernel |
| `x_sf` | packed scale factors for `x` | symmetric global buffer before kernel |
| `topk_idx` | selected expert IDs or `-1` for invalid routes | symmetric global buffer before kernel |
| `topk_weights` | route weights | copied into L1 route state during dispatch |
| `l1_acts` / `l1_acts_sf` | Linear1 input payload and SF | local L1 ring slots |
| `l2_acts` / `l2_acts_sf` | Linear2 input activation and SF | local L2 ring slots |
| `combine_token_buffer` | BF16 route outputs before final sum | token-owner rank symmetric buffer |
| `y` | final local token output | ordinary output tensor |

`topk_idx` stores expert IDs, not combine-slot indices. During final combine,
the code uses `topk_idx[token, lane] >= 0` as a validity mask; the actual
combine slot is the lane / bit position.

## Capacity And Balance

`num_max_tokens_per_rank` is a launch-time capacity bound for per-rank
symmetric-buffer views. It is not computed dynamically by the fused kernel.
The exact production policy for choosing this capacity is not pinned in these
notes.

Implications:

- real routing imbalance must fit within configured capacity or be handled by
  upstream routing / serving policy;
- capacity overprovisioning increases symmetric-buffer footprint;
- pathological imbalance can reduce overlap because one local expert or wave
  can become the tail.

## Expert Count Names

The public code has variables such as `shared_expert_count`. Do not read this
as the model-architecture "shared expert" concept. In this context it refers
to count/state sharing inside the kernel protocol, not a separate modeling
expert that all tokens always visit.

## Pool And Ring Concepts

Use these names consistently:

| Term | Meaning |
|---|---|
| pool | full logical token sequence for local expert work |
| pool token | one routed token occurrence in that full logical sequence |
| ring token / ring slot | reusable physical storage slot for a subset of pool tokens |
| ring block | group of ring slots that share producer-consumer generation state |
| generation | reuse epoch for the same physical ring slot |

The pool is logical and monotonic; the ring is physical and reusable:

```text
pool_token_idx:
  0, 1, 2, 3, 4, 5, ...

physical ring slot for num_ring_tokens = 4:
  0, 1, 2, 3, 0, 1, ...
```

This is why code patterns such as
`l1_token_buffer[pool_token_idx % num_ring_tokens]` should be read as:

```text
logical pool position
  -> physical reusable ring slot
```

## Producer / Consumer Counters

The ring protocol is a producer-consumer protocol with generation counters.
A simplified model:

```text
producer for ring slot s, generation g:
  wait until slot s is empty for generation g
  write payload into slot s
  publish full_count[s] for generation g

consumer for ring slot s, generation g:
  wait until full_count[s] reaches expected value
  consume payload
  publish empty_count[s] for generation g + 1
```

Important property:

```text
slot 2 can be reused for the next generation
even if slots 0, 1, and 3 are still being consumed
```

as long as the generation counter for slot 2 proves that its previous consumer
has released it.

## Cross-Stage Protocol Edges

| Edge | Producer publishes | Consumer waits on | Notes |
|---|---|---|---|
| dispatch -> Linear1 | `l1_acts`, `l1_acts_sf`, route weight, source metadata, `l1_full_count` | Linear1 scheduler / GEMM body | dispatch-specific details live in dispatch note |
| Linear1 -> activation | FP32 accumulator tile in TMEM | activation epilogue | TMEM full/empty barriers, not global ring counters |
| activation -> Linear2 | `l2_acts`, `l2_acts_sf`, `l2_full_count` | Linear2 scheduler / GEMM body | L2 here means Linear2 input ring, not L2 cache |
| Linear2 -> combine buffer | remote BF16 route-output slot | final combine after barrier | no atomic add needed per route slot |
| write-back -> final combine | pre-combine NVLink barrier | final local reducer | phase-level barrier, not per-token |

This table is the main map for deciding which concrete note owns a detail.

## Resource Overlap Model

The public SM100 kernel partitions work into persistent roles rather than
launching a new CUDA kernel per logical phase. The exact policy is selected by
DeepGEMM heuristics, but the mental model is:

```text
dispatch role:
  remote input / metadata movement, ring-slot publication

GEMM role:
  TMA A/B/SFA/SFB loads, shared-memory K-tile staging, UMMA accumulation

epilogue role:
  TMEM reads, activation / cast / pack, L2 ring publish or combine write-back

combine role:
  after pre-combine barrier, local top-k slot reduction
```

Do not interpret warp counts as direct power or Tensor Core allocation. Warp
roles are scheduling / orchestration choices; UMMA and TMA can drive dedicated
hardware mechanisms after a relatively small number of threads issue the right
operations.

## What Belongs Elsewhere

- Dispatch route ordering, metadata push, and L1 pull:
  [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md)
- Wave windows, persistent workers, bubbles, and source examples:
  [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md)
- Scale-factor layout, UTCCP, and UMMA consumption:
  [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) and
  [`deepseek-v4-megamoe-quantization.md`](deepseek-v4-megamoe-quantization.md)
- Symmetric-memory implementation details:
  [`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-cuda-symmetric-memory)

## Reading Checklist

When adding new MegaMoE notes, classify each fact before writing it down:

1. Is it algorithm semantics?
2. Is it distributed EP ownership?
3. Is it a runtime protocol object?
4. Is it scheduling policy?
5. Is it GEMM / activation / combine stage logic?
6. Is it CUDA / hardware background?

If the answer is "runtime protocol object", put it here or link here.

## Open Questions

- What production policy chooses `num_max_tokens_per_rank` under real serving
  or training load?
- How should this protocol change for backward kernels, especially dW / dX
  reductions?
- Which parts of this protocol survive if cross-node RDMA / IB enters the EP
  path?
