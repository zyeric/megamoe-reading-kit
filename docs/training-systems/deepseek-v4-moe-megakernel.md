# DeepSeek-V4 MoE Megakernel Reading Plan

Status: focused deep dive

Context metadata:

- Topic: model-to-kernel lowering map for DeepSeek-V4 MoE / MegaMoE.
- Layer tags: `algorithm`, `distributed-ep`, `runtime-protocol`,
  `scheduling`, `cuda-model`.
- Owns: research framing, lowering layers, reading roadmap, top-level dtype /
  stage map.
- Does not own: detailed dispatch mechanics, scale-factor ABI, GEMM pipeline,
  or symmetric-memory backend details.
- Agent entry: read this after
  [`deepseek-v4-megamoe-context-map.md`](deepseek-v4-megamoe-context-map.md)
  when the task asks "why does this megakernel exist?".

Primary source:

- DeepSeek-V4 technical report: [`../sources/2606.19348v1-deepseek-v4.pdf`](../sources/2606.19348v1-deepseek-v4.pdf)
- Cross-area entrypoint: [`../model-architecture/deepseek-v4.md`](../model-architecture/deepseek-v4.md)
- Hardware background: [`gpu-hardware-notes/notes/nvidia-gpu-generations.md`](https://zyeric.github.io/gpu-hardware-notes/notes/nvidia-gpu-generations.md)
- Runtime protocol notes:
  [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
- Dispatch notes: [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md)
- Scheduling notes: [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md)
- Quantization / scale-factor notes:
  [`deepseek-v4-megamoe-quantization.md`](deepseek-v4-megamoe-quantization.md)
- Activation / SwiGLU epilogue notes:
  [`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md)
- Linear1 / Linear2 GEMM notes:
  [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md)
- Combine notes:
  [`deepseek-v4-megamoe-combine.md`](deepseek-v4-megamoe-combine.md)

Related implementation sources to inspect:

- DeepSeek MegaMoE, when available from the V4 release artifacts.
- DeepEP: <https://github.com/deepseek-ai/DeepEP>
- DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>
- CUTLASS / CuTe: <https://github.com/NVIDIA/cutlass>
- Triton: <https://github.com/triton-lang/triton>

## Focus

Read DeepSeek-V4 through one narrow systems question:

> How does the V4 fused MoE megakernel change the execution and correctness
> contract of expert-parallel MoE training?

This note intentionally does not try to cover the full V4 stack at once. CSA /
HCA sparse attention, million-token context, FP4 QAT, inference, and
post-training should stay in separate notes until the MoE kernel path is clear.

Current scope decisions:

- Read the Mega MoE path first. Do not read every DeepGEMM feature linearly.
- Start with Hopper / SM90. Treat SM100 / Blackwell as a later delta after the
  SM90 execution model is understood.
- Work without GPU profiling for now. Prioritize source maps, execution
  contracts, static reasoning, toy semantic checks, and benchmark caveat
  analysis.
- Defer MQA scoring and HyperConnection code paths. They remain useful later
  for sparse-attention and HyperConnection notes, but they are out of scope for
  this MoE megakernel pass.
- Explain the megakernel from model architecture pressure first: why the V4 MoE
  workload creates enough dispatch, communication, imbalance, metadata, and
  small / grouped GEMM overhead to justify fusing across boundaries.

## Why This Subtopic

- It is close to GPU execution rather than trainer glue.
- It connects directly to training throughput, communication overlap, and MoE
  scaling.
- It exposes correctness and determinism boundaries that matter for training
  verification.
- It uses public DeepSeek-style artifacts and SM90-era implementation patterns
  as concrete reading anchors.
- It can produce useful notes without immediate access to B200 / GB200 /
  Rubin-class hardware.

## Questions

### Architecture Motivation

- Which DeepSeek-V4 MoE architecture choices make the expert path a first-order
  system bottleneck?
- What are the routing, top-k, expert granularity, token-count, and sequence
  shape assumptions that create pressure for a fused Mega MoE path?
- Why is a separate dispatch + grouped GEMM + combine pipeline insufficient or
  less attractive under the reported V4 workload?
- Which parts of the motivation are model/workload-driven, and which parts are
  SM90 hardware-driven?

### MoE Semantics

- What is the exact forward contract for routing, dispatch, expert computation,
  top-k weighting, and combine?
- Which tensor layouts and token-order assumptions are required by the fused
  kernel?
- Which parts of the contract are inherited from DeepSeek-V3-style MoE, and
  which parts are new in V4?

### Megakernel Execution

- Which stages are fused: dispatch, communication, memory movement, grouped
  GEMM, combine, or backward reductions?
- What is the first code path to read end-to-end, and what public test or
  wrapper defines its executable contract?
- Is the kernel persistent, and how is work scheduled across SMs?
- How are expert imbalance, token skew, padding, and empty experts handled?
- Where are TMA, WGMMA, warp specialization, or similar SM90 mechanisms used or
  implied?
- Which parts remain outside the megakernel, and why?

### Communication And Overlap

- How does the V4 path relate to DeepEP dispatch / combine?
- What communication primitive is assumed: NCCL, NVSHMEM, custom GPU-initiated
  communication, or a hybrid?
- What is being overlapped with what: all-to-all, memory copy, GEMM, combine,
  or pipeline bubbles?
- What benchmark would fairly compare separate DeepEP + grouped GEMM against a
  fused megakernel?

### Determinism And Verification

- Where can token order affect expert outputs or gradients?
- Which backward paths rely on atomics, split-k reductions, multi-stage
  reductions, or non-stable accumulation order?
- What does batch invariance mean for MoE forward and backward?
- Which buffers must be isolated or ordered to make same-layout replay exact?
- What minimal tests would catch token-order, routing, combine, or expert-grad
  drift?

## Working Lowering Map

Use a compiler-style lowering view while reading MegaMoE. The point is to keep
the mathematical MoE semantics stable while progressively rewriting the
execution schedule, intermediate representation, and hardware mapping.

This is not always a strict one-to-one mapping. A single algorithmic step can
lower into multiple kernel patterns, and one kernel pattern can cover multiple
algorithmic steps after fusion. The useful invariant is:

```text
same MoE semantics
  different schedule
  different intermediate buffers
  different synchronization and memory hierarchy use
```

### 1. Math / Algorithm Spec

Forward semantics:

```text
routes_i = topk(router(x_i))

FFN_e(x) = W2_e * act(W1_e * x)

y_i = sum_{j in topk(i)} gate_weight_{i,j} * FFN_{expert_{i,j}}(x_i)
```

Questions at this layer:

- What exactly must be numerically equivalent to the reference?
- What is the routing contract: top-k, expert IDs, gate weights, token order?
- Where are precision changes allowed: input, expert weights, activation,
  accumulation, output?
- What backward semantics are required for `x`, expert weights, router, and
  gate weights?

### 2. Naive Torch Reference

A reference implementation can be read as:

```python
routes = router_topk(x)
expert_inputs = dispatch_by_expert(x, routes)
hidden = grouped_expert_linear1(expert_inputs, w1)
hidden = activation(hidden)
expert_outputs = grouped_expert_linear2(hidden, w2)
y = combine_by_token(expert_outputs, routes.weights)
```

This layer makes the semantic contract easy to test, but it materializes
intermediate tensors and usually launches separate operations for dispatch,
expert compute, activation, and combine.

Questions at this layer:

- What are the reference intermediate tensors and shapes?
- Which operations are separate in the reference?
- Which ordering choices are observable in outputs or gradients?
- Which unit tests define correctness independent of performance?

### 3. Distributed EP Reference

Before jumping to MegaMoE, add an expert-parallel reference layer. This layer
keeps the implementation understandable, but makes rank ownership and
communication explicit.

Sketch:

```python
# Each rank starts with local tokens.
routes = router_topk(local_x)

# Decide which rank owns each selected expert.
send_metadata = build_ep_dispatch_metadata(routes, expert_to_rank)
send_tokens = pack_tokens_by_destination_rank(local_x, routes)

# Cross-rank all-to-all / dispatch.
recv_tokens, recv_metadata = ep_dispatch_all_to_all(send_tokens, send_metadata)

# Each rank computes only the experts it owns.
expert_inputs = group_by_local_expert(recv_tokens, recv_metadata)
hidden = grouped_expert_linear1(expert_inputs, local_w1)
hidden = activation(hidden)
expert_outputs = grouped_expert_linear2(hidden, local_w2)

# Send per-route outputs back to the token-owner ranks.
send_outputs = pack_outputs_by_source_token_rank(expert_outputs, recv_metadata)
route_outputs = ep_combine_all_to_all(send_outputs)

# Reconstruct local token outputs.
y = combine_by_token(route_outputs, routes.weights)
```

This layer is the right baseline for understanding why MegaMoE is not merely a
faster grouped GEMM. The workload boundary is:

```text
local tokens
  -> route metadata
  -> cross-rank dispatch
  -> local expert compute
  -> cross-rank combine
  -> local token outputs
```

Questions at this layer:

- Which rank owns each expert?
- Which tensors are token-owned and which tensors are expert-owned?
- What metadata must travel with each routed token?
- Where does all-to-all / dispatch happen, and where does combine happen?
- Which operations are separated by communication boundaries?
- Which intermediate buffers are full materializations rather than streaming or
  ring-buffered slots?
- What would be a fair DeepEP + grouped GEMM baseline?

### 4. Execution Rewrite

MegaMoE-style execution changes the schedule and intermediate representation:

| Reference logic | Fused execution rewrite |
|---|---|
| EP dispatch all routed tokens into full expert input tensors | dispatch into reusable ring-buffer slots |
| one grouped GEMM boundary for Linear1 | persistent tiled Linear1 over expert blocks / waves |
| explicit activation tensor | activation fused into Linear1 epilogue |
| full intermediate materialization | ring-buffered Linear2 input slots |
| one grouped GEMM boundary for Linear2 | persistent tiled Linear2 over expert blocks / waves |
| separate EP combine op | combine fused after per-route outputs are ready |
| operation-level synchronization | producer-consumer counters and peer-visible barriers |

Questions at this layer:

- Which torch-level boundaries are removed by fusion?
- Which intermediate tensors disappear or become ring-buffer slots?
- Which stages can overlap: dispatch, TMA load, GEMM, epilogue, combine?
- Which synchronization points are still semantically required?

### Forward Dtype Map

Quick dtype map for the public DeepGEMM SM100 FP8/FP4 MegaMoE forward path.
Read the quantization note for scale-factor layout, block-scaled matmul
semantics, and the V3 / V4 quantization-recipe boundary:
[`deepseek-v4-megamoe-quantization.md`](deepseek-v4-megamoe-quantization.md).

| Stage | Inputs | Outputs | Where conversion happens |
|---|---|---|---|
| Model semantic input | BF16 hidden states from previous block | BF16 hidden states | Outside MegaMoE kernel semantics |
| Caller-side input quantization | BF16 hidden states | `x` FP8 E4M3 payload + `x_sf` packed UE8M0 scale factors | Outside `fp8_fp4_mega_moe`; test uses `per_token_cast_to_fp8(..., gran_k=32)` |
| Dispatch pull | remote `x` FP8 + `x_sf`; `topk_idx`; `topk_weights` FP32 | local `l1_acts` FP8 + `l1_acts_sf`; local per-route top-k weight | Inside MegaMoE kernel, through symmetric-memory pull into L1 ring slots |
| Linear1 GEMM | `l1_acts` FP8 + SFA; L1 weights FP4 + SFB | FP32 accumulator in TMEM | Block-scaled UMMA consumes payloads and scale factors |
| Linear1 epilogue / SwiGLU | FP32 accumulator; per-route top-k weight FP32 | `l2_acts` FP8 E4M3 + `l2_acts_sf` packed UE8M0 | Inside MegaMoE kernel: gate/up, clamp/SwiGLU, top-k weight, amax, scale, FP8 quantization |
| Linear2 GEMM | `l2_acts` FP8 + SFA; L2 weights FP4 + SFB | FP32 accumulator in TMEM | Block-scaled UMMA consumes payloads and scale factors |
| Linear2 epilogue / combine write-back | FP32 accumulator plus source-token metadata | BF16 per-route output in remote combine buffer | Inside MegaMoE kernel; no new FP8 output scale is produced |
| Final combine reduction | BF16 per-route outputs for each token/top-k slot | final `y` BF16 | Sum preweighted route outputs; top-k weight has already been applied in Linear1 epilogue |

Two boundaries are easy to confuse:

```text
initial x quantization:
  BF16 -> FP8 + x_sf happens before the MegaMoE kernel

Linear1 intermediate quantization:
  FP32 accumulator -> FP8 l2_acts + l2_acts_sf happens inside the kernel
```

So the short dtype chain is:

```text
BF16 hidden state
  -> FP8 x + x_sf
  -> Linear1 FP32 accumulator
  -> FP8 l2_acts + l2_acts_sf
  -> Linear2 FP32 accumulator
  -> BF16 combine buffer / y
```

### 5. Kernel-Pattern Layer

Map the rewritten execution to implementation patterns:

| Algorithm / execution need | Kernel pattern |
|---|---|
| small or imbalanced expert token counts | persistent kernel with device-side scheduler |
| many experts with insufficient per-expert blocks | wave scheduling across expert groups |
| limited workspace for routed tokens and intermediates | ring buffers |
| dispatch / compute / combine overlap | producer-consumer pipeline |
| readiness between stages | `full_count` / `empty_count` counters and spin-wait |
| local expert GEMM | tiled tensor-core GEMM |
| activation / scaling / quantization | fused epilogue |
| preweighted route-output reconstruction | gather / sum combine |

Questions at this layer:

- What is the logical work unit returned by the scheduler?
- What does one ring-buffer slot represent?
- Which counters make a slot safe to produce or consume?
- How does the wave size trade off SM utilization, locality, and ring capacity?

### 6. CUDA Programming Model

Lower the kernel patterns to CUDA-visible objects:

| Kernel pattern | CUDA programming-model objects |
|---|---|
| persistent kernel | long-running CTAs, loops, device-side state |
| tiled GEMM | CTA tiles, warpgroup roles, shared-memory stages |
| producer-consumer pipeline | atomics, barriers, acquire / release-style memory ordering |
| ring buffer | global-memory workspace plus modulo indexing |
| dispatch / combine | warps reading route metadata and writing local / peer buffers |
| epilogue | warp-level stores, vector ops, output layout conversion |

Questions at this layer:

- How many CTAs / warps / warpgroups are launched?
- Which warps own dispatch, TMA loads, MMA, epilogue, and combine?
- What is stored in registers, shared memory, TMEM, L2 / HBM, and peer memory?
- What barrier scope is required: warp, CTA, cluster, device, or peer rank?

### 7. ISA / Hardware Layer

The final lowering maps CUDA code and generated templates to hardware
primitives:

| CUDA / kernel need | ISA / hardware primitive |
|---|---|
| tensor tile movement | TMA or ordinary global / shared-memory load-store instructions |
| matrix multiply | MMA / WGMMA / UMMA tensor-core instructions |
| accumulator state | registers on SM90-style paths, TMEM on SM100-style paths |
| producer-consumer counters | atomics, scoped memory operations, L2-visible loads / stores |
| peer communication | NVLink / NVSwitch / PCIe peer memory, or a separate RDMA / network stack if cross-node |
| locality and reuse | shared memory, L1, L2, HBM bandwidth |

Questions at this layer:

- Which architecture path is being read: SM90, SM100, or both?
- Which low-level primitives are required versus optional optimizations?
- What limits occupancy: registers, shared memory, TMEM, barriers, or CTA
  shape?
- Which roofline is relevant: tensor-core compute, HBM, L2, NVLink, network, or
  scheduling overhead?

## Dispatch Lowering

Cross-stage runtime objects are tracked in a separate note:
[`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md).

Dispatch is now tracked in a separate note:
[`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md).

The short version:

```text
Distributed EP reference:
  pack full token tensors
  all-to-all dispatch
  materialize expert input tensors

MegaMoE dispatch rewrite:
  use already-computed top-k route state
  push / read per-source-rank expert counts
  pull token payloads into local L1 ring slots
  record TokenSrcMetadata for later combine write-back
  publish l1_full_count for Linear1 consumers
```

Read the runtime protocol note for symmetric-buffer addressability, pool /
ring-buffer indexing, counters, and capacity. Read the dispatch note for
metadata push, source-rank order, token pull parallelism, `TokenSrcMetadata`,
`l1_full_count`, spin-wait in dispatch context, and the EP8 worked sketch.

## Execution Scheduling

MegaMoE execution scheduling is tracked in a separate note:
[`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md).

The short version:

```text
wave:
  high-level scheduling window over a subset of local experts

pool block:
  logical BLOCK_M token block in the expert-token pool

ring block:
  physical reusable BLOCK_M slot in the L1/L2 activation ring buffer
```

Read the scheduling note for the relationship between waves, pool blocks, ring
blocks, persistent workers, pipeline counters, wave-boundary behavior, and why
wave scheduling helps small or imbalanced expert workloads.

## Activation Epilogue

Linear1 activation / SwiGLU is now tracked in a separate note:
[`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md).

The short version:

```text
Linear1 FP32 TMEM accumulator
  -> BF16-rounded gate/up
  -> clamp + SiLU(gate) * up * top-k weight
  -> amax / UE8M0 scale
  -> FP8 l2_acts + l2_acts_sf
  -> l2_full_count for Linear2
```

Read the activation note for the fused path versus TileLang baseline boundary,
gate/up TMEM reads, precision path, scale-factor stores, and the producer /
consumer handoff from Linear1 to Linear2.

## Linear1 / Linear2 GEMM

Linear1 and Linear2 share the same TMA / UMMA / TMEM GEMM body, tracked in:
[`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md).

The short version:

```text
scheduler block phase
  -> choose L1 or L2 A/B/SF tensor maps
  -> TMA load A/SFA and B/SFB into shared-memory K stages
  -> UTCCP move scale metadata for block-scaled UMMA
  -> UMMA accumulates complete FP32 C tile in TMEM
  -> phase-specific epilogue consumes the C tile
```

Read the GEMM note for the shared Linear1/Linear2 tensor maps, K-tile pipeline,
scale-factor movement, UMMA accumulation order, TMEM handoff, and forward
determinism boundary. Linear2 remote write-back and final top-k slot summation
should stay in the later combine note.

## Combine

Linear2 route-output write-back and final token reconstruction are tracked in:
[`deepseek-v4-megamoe-combine.md`](deepseek-v4-megamoe-combine.md).

The short version:

```text
Linear2 epilogue
  -> BF16 route output
  -> remote combine_token_buffer[token, topk_slot]
  -> pre-combine cross-rank barrier
  -> local top-k slot sum
  -> y[token]
```

Read the combine note for `TokenSrcMetadata`, remote symmetric-buffer writes,
the pre-combine NVLink barrier, chunked local reduction, and why the final
combine sums preweighted route outputs rather than multiplying `topk_weights`
again.

## Forward Path Remaining Roadmap

The first forward-path pass should close the remaining SM100 FP8/FP4 MegaMoE
mechanisms before moving to backward, DeepEP baselines, or production-training
claims.

1. Quantization / scale-factor path
   - Track `x_sf`, `l1_acts_sf`, `l2_acts_sf`, weight scale factors, and their
     layouts.
   - Explain where amax, scale, inverse scale, FP8 quantization, and SF stores
     happen.
   - Separate input / weight scaling from Linear1 epilogue output
     quantization.
2. SwiGLU / activation epilogue
   - Decode how Linear1 output is interpreted as gate / up.
   - Track clamp, SiLU, multiply, top-k weight application, and L2 ring
     production.
   - Explain why activation is an epilogue role rather than a standalone stage.
3. GEMM / UMMA determinism
   - Trace TMA load -> shared memory -> UMMA -> TMEM -> epilogue.
   - Identify deterministic block order, fixed reduction order, and any atomic
     / counter boundaries.
   - Separate forward determinism from backward gradient-reduction risks.
4. Combine flow and overlap
   - Trace Linear2 epilogue remote write-back into combine buffers.
   - Explain the pre-combine peer-visible barrier and final top-k reduction.
   - Verify whether overlap is combine reduction with dispatch workspace
     cleanup, rather than combine with the initial dispatch pull.
5. Boundary / open questions
   - Mark public MegaMoE forward coverage versus missing training backward.
   - Keep cross-node RDMA / DeepEP training communication separate from this
     symmetric-memory / NVLink-oriented path.
   - Record SM90 versus SM100 deltas only after the public SM100 path is clear.

## Deliverables

- Model-architecture motivation note: why V4 needs a Mega MoE-style path.
- Execution timeline for V4-style MoE forward and backward.
- MoE forward/backward correctness checklist.
- Source map for one Mega MoE code path: public API / test, Python wrapper,
  generated or JIT boundary, CUDA / template kernel, communication buffer
  setup, and correctness / benchmark harness.
- Mechanism note comparing V3 DeepEP-style execution with V4 fused MoE
  execution.
- Determinism note covering token order, atomics, split-k, expert gradients, and
  communication overlap.
- Optional toy prototype in a scratch area or separate repo:
  - PyTorch MoE forward/backward that intentionally varies token order and
    accumulation order.
  - Small Triton/CUDA kernel only if it helps explain execution, not as a
    performance claim.

## Reading Stages

### Stage 1: V4 MoE Architecture Motivation

- Extract the V4 MoE architecture choices that create pressure for a fused
  kernel path.
- Separate model/workload motivation from hardware-specific optimization.
- Record the minimum shape, routing, precision, and expert-parallel assumptions
  needed to understand the Mega MoE path.

### Stage 2: V4 Report Extraction

- Extract every V4 claim about MoE kernels, MegaMoE, determinism, batch
  invariance, expert parallelism, and communication overlap.
- Record each claim with: mechanism, evidence, system implication, benchmark
  caveat, and open question.

### Stage 3: One DeepGEMM / Mega MoE Code Path

- Pick one public Mega MoE forward path and follow it end-to-end.
- Read the test or example first to define the executable contract: inputs,
  shapes, dtypes, layouts, routing metadata, expected outputs, correctness
  checks, and benchmark assumptions.
- Use agent-assisted source indexing for call graph and code references, but
  keep the human focus on contracts, execution timeline, and design boundaries.
- Do not enter MQA scoring, HyperConnection, or unrelated dense GEMM paths in
  this stage.

### Stage 4: V3 / DeepEP Baseline

- Reconstruct the V3-style MoE execution boundary: routing, DeepEP dispatch,
  grouped GEMM, combine, DualPipe overlap, and FP8 assumptions.
- Identify which boundary V4 tries to collapse with the megakernel.

### Stage 5: SM90 Execution Model

- Review the minimum Hopper / SM90 concepts needed for the fused path:
  Tensor Core, TMA, WGMMA, shared-memory pipeline, persistent kernels, and SM
  occupancy tradeoffs.
- Keep SM100 / Blackwell / Rubin as a later delta, not the primary target.

### Stage 6: Determinism Model

- Map each MoE stage to deterministic and non-deterministic risks.
- Define testable contracts for same-layout replay, token-order stability,
  expert-gradient accumulation, and communication-buffer isolation.

### Stage 7: Public Writeup

- Draft a focused writeup:
  `DeepSeek-V4 MoE Megakernel: Routing Semantics, GPU Execution, and Training
  Determinism`.
- Keep claims source-linked and mark speculation explicitly.

## Non-Goals

- Do not build a full DeepSeek-V4 survey in this note.
- Do not implement a production megakernel without access to suitable GPU
  hardware and benchmarks.
- Do not profile performance until suitable GPU access is available.
- Do not read MQA scoring or HyperConnection in this pass.
- Do not make SM100 / Blackwell claims before the SM90 path is understood.
- Do not compare framework-level RL systems here.
- Do not promote any reusable rule into `contexts/*/knowledge/` until it is
  backed by a concrete source or experiment.
