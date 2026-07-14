# DeepSeek-V4 MegaMoE Reading Guide

Status: human-first top-down guide

Context metadata:

- Topic: shallow-to-deep human reading path for MegaMoE.
- Layer tags: `algorithm`, `distributed-ep`, `runtime-protocol`,
  `scheduling`, `cuda-model`.
- Owns: top-down story, reading tracks, text diagrams, and memory refresh path.
- Does not own: claim evidence ledger, exact code provenance, or stage-specific
  source details.
- Agent entry: read when the task asks for a presentation, public writeup, or
  re-entry path after time away.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-context-map.md`](deepseek-v4-megamoe-context-map.md)
  - agent routing and note ownership.
- [`deepseek-v4-megamoe-glossary.md`](deepseek-v4-megamoe-glossary.md) -
  layer-aware term definitions.
- [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md)
  - evidence levels and open questions.

## Purpose

This is the document to read after coming back to the topic weeks later. It is
not the deepest source note. Its job is to rebuild the whole mental model from
shallow to deep before jumping into code-backed details.

## Reading Tracks

| Track | Time | Read |
|---|---:|---|
| Refresh | 10 minutes | this guide, then the glossary table |
| Talk prep | 1 hour | this guide, `deepseek-v4-megamoe-talk.html`, claims index |
| Code reading | half day | context map, source snapshot, runtime protocol, scheduling, then stage notes |
| Deep dive | multi-day | all source notes plus hardware appendices and upstream code revalidation |

## One-Sentence Thesis

MegaMoE is best understood as an execution rewrite of expert-parallel MoE:
instead of separate dispatch, grouped GEMM, activation, send-back, and combine
phases, the public DeepGEMM path fuses these responsibilities into a persistent
GPU-side protocol using peer-visible buffers, ring slots, counters, and
Blackwell GEMM machinery.

## The Lowering Path

Read the system as progressive lowering:

```text
MoE math
  -> naive Torch semantics
  -> distributed expert parallel reference
  -> fused MegaMoE execution rewrite
  -> CUDA kernel patterns
  -> PTX / ISA / hardware mechanisms
```

At each layer, the semantic goal stays the same but the representation changes.

| Layer | Main question | Read |
|---|---|---|
| MoE math | What output should this compute? | `deepseek-v4-moe-megakernel.md` |
| Naive Torch | What would the unoptimized reference do? | `deepseek-v4-moe-megakernel.md` |
| Distributed EP | Which rank owns each expert route? | `deepseek-v4-moe-megakernel.md` |
| Runtime protocol | How is ownership represented in one fused kernel? | `deepseek-v4-megamoe-runtime-protocol.md` |
| Scheduling | Which work is exposed to resident workers? | `deepseek-v4-megamoe-scheduling.md` |
| Stage mechanics | How does each pipeline step work? | dispatch / quantization / activation / GEMM / combine notes |
| Hardware | Which GPU concepts make this possible? | `https://zyeric.github.io/gpu-hardware-notes/` |

## Visual Model 1: MoE Semantics

```text
token hidden states
  -> router selects top-k experts
  -> each selected expert runs an FFN route
  -> route outputs are weighted
  -> weighted route outputs are summed back per token
```

This layer says nothing about ranks, NVLink, ring buffers, or tensor cores.

## Visual Model 2: Distributed EP Reference

```text
token-owner rank
  -> sends selected routes to expert-owner ranks
  -> expert-owner ranks run local expert FFNs
  -> route outputs go back to token-owner rank
  -> token-owner rank combines top-k route outputs
```

This is the conceptual baseline. It can be implemented by collectives,
point-to-point communication, DeepEP-style all-to-all, or a fused peer-memory
protocol. MegaMoE changes the implementation boundary, not the route ownership
semantics.

## Visual Model 3: MegaMoE Fused Forward

```text
before fused kernel:
  BF16 hidden states are quantized to x + x_sf
  router/top-k state is already computed
  symmetric-buffer views are prepared

inside fused kernel:
  dispatch pull:
    remote x/x_sf/topk state -> local L1 ring slots
  Linear1:
    L1 ring slot + L1 weight -> FP32 TMEM accumulator
  activation:
    gate/up + top-k weight -> FP8 L2 ring slot + l2_acts_sf
  Linear2:
    L2 ring slot + L2 weight -> FP32 TMEM accumulator
  write-back:
    route output -> remote token-owner combine slot
  final combine:
    barrier -> local fixed-order top-k-slot reduction -> y
```

The important shift is that dispatch and combine are no longer clean external
communication phases. They are part of a persistent kernel protocol.

## Visual Model 4: Pool Versus Ring

```text
logical pool token order:
  0, 1, 2, 3, 4, 5, 6, 7, ...

physical ring slots if num_ring_tokens = 4:
  0, 1, 2, 3, 0, 1, 2, 3, ...

slot reuse is guarded by generation counters:
  producer waits for empty
  producer writes payload
  producer publishes full
  consumer waits for full
  consumer consumes payload
  consumer publishes empty
```

The pool is the logical work sequence. The ring is bounded physical workspace.
Do not confuse a ring block with a wave or a GEMM tile.

## Visual Model 5: Memory / Hardware Path

```text
peer-visible global memory:
  x, x_sf, topk_idx, topk_weights, combine_token_buffer

local global ring buffers:
  l1_acts / l1_acts_sf
  l2_acts / l2_acts_sf

shared memory:
  staged A/B/SFA/SFB tiles for GEMM

TMEM:
  FP32 accumulator tiles for UMMA and epilogue handoff

registers / scalar lanes:
  epilogue math, amax, scale, combine reduction chunks
```

The difficult part of reading the kernel is mapping each variable to one of
these storage roles before reasoning about performance or correctness.

## Suggested First-Pass Reading Order

1. Read [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md) until
   the lowering layers are clear.
2. Read [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
   before dispatch. This prevents symmetric memory, pool/ring, and counters
   from being mistaken as dispatch-only concepts.
3. Read [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md)
   to separate wave, pool block, ring block, GEMM tile, and K-stage.
4. Read the stage notes in execution order:
   dispatch, quantization, GEMM, activation, GEMM again, combine.
5. Use the hardware notes only when blocked by a term:
   CTA, warpgroup, TMA, TMEM, UMMA, UTCCP, cache, peer memory.
6. End with [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md)
   to check what is code-backed versus still open.

## What To Remember

- Public DeepGEMM MegaMoE notes here are about the SM100 FP8/FP4 forward path.
- Router/top-k is outside the fused kernel in the public path described here.
- Input quantization is outside the fused kernel; activation re-quantization
  for Linear2 happens inside the activation epilogue.
- Symmetric memory gives peer-addressable buffers and a setup/runtime model. It
  is not by itself a full cross-node training communication stack.
- `wave`, `ring block`, `pool block`, `GEMM tile`, and `top-k slot` are
  different units.
- Top-k weight is applied before Linear2 in the public FP8/FP4 path, so final
  combine sums preweighted route outputs.
- The strongest durable idea is the lowering pattern: move EP ownership,
  memory movement, scheduling, and compute into a GPU-side protocol when the
  workload has enough small-batch / imbalance pressure.

## What To Revalidate Later

Before reusing these notes for a new chip, new DeepGEMM commit, or training
kernel:

1. Re-pin the upstream code SHA in
   [`deepseek-v4-megamoe-source-snapshot.md`](deepseek-v4-megamoe-source-snapshot.md).
2. Re-run the claims in
   [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md).
3. Check whether backward kernels exist and whether they preserve the same
   fusion boundary.
4. Check whether SM90 / SM100 / Rubin paths change the scale-factor and GEMM
   pipeline assumptions.
5. Check whether cross-node EP uses DeepEP / RDMA outside this fused kernel
   boundary.
