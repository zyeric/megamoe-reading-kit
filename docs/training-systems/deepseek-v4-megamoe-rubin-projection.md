# Rubin Opportunity And Challenge Map For MegaMoE

Date: 2026-08-03

Status: hardware-backed projection from the public SM100 MegaMoE forward path;
no public Rubin MegaMoE kernel, CUDA tuning guide, SASS, profile, backward path,
or measured result is pinned

Context metadata:

- Topic: how public Rubin mechanisms could change the current MegaMoE
  execution boundary.
- Layer tags: `distributed-ep`, `runtime-protocol`, `scheduling`,
  `stage-gemm`, `stage-combine`, `hardware`, `evidence`.
- Owns: cross-stage Rubin opportunity/challenge mapping, EP scaling model, and
  port-validation questions.
- Does not own: generic Rubin specifications or claims about an implemented
  Rubin path.
- Agent entry: read after the current SM100 runtime/stage notes when projecting
  the design to Rubin.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - current symmetric buffers, rings, counters, and barriers;
- [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md)
  - waves, persistent workers, tails, and imbalance;
- [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) - current
  SM100 TMA/UMMA/TMEM path;
- [`deepseek-v4-megamoe-combine.md`](deepseek-v4-megamoe-combine.md) - remote
  route write-back and whole-phase barrier;
- [GPU generation notes](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-nvidia-gpu-generations)
  - owner of generic Rubin facts and sources.

## 1. Evidence Boundary

The hardware inputs are NVIDIA's public
[Rubin GPU architecture overview](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/),
[Rubin platform overview](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/),
and [Vera Rubin NVL72 product page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/).
They document HBM4, doubled Tensor Core K-dimension processing, inline TMA
descriptor update, fine-grained dependent triggering, counted writes, NVLink
6, and rack-scale figures.

The implementation input is the public DeepGEMM SM100 FP8/FP4 forward path
already pinned in this kit. It is not evidence that the same ABI, instruction
sequence, resource partition, or fusion boundary exists on Rubin.

Use the following labels throughout:

| Label | Meaning here |
|---|---|
| Hardware fact | Explicitly described in public NVIDIA Rubin material |
| Current-path fact | Visible in the pinned public SM100 DeepGEMM path |
| Projection | A plausible mapping between the two that needs implementation and measurement |
| Open | Public material is insufficient to choose or prove the design |

## 2. Current SM100 Baseline

The fused forward path in these notes is:

```text
router/top-k and input quantization outside kernel
  -> peer pull into local L1 ring
  -> Linear1 TMA / UMMA / TMEM GEMM
  -> SwiGLU + top-k weight + amax + FP8 requant into L2 ring
  -> Linear2 TMA / UMMA / TMEM GEMM
  -> remote BF16 write into token-owner top-k slot
  -> whole-phase NVLink barrier
  -> fixed slot-order local combine
```

The durable abstraction is not any one Blackwell instruction. It is a
GPU-resident protocol that makes route ownership, reusable storage,
producer-consumer readiness, compute, and write-back explicit.

## 3. EP Scaling: Mean Expert Load Is Not Set By EP Degree

Let one data-parallel replica present an MoE layer with:

```text
T = tokens in this DP replica
k = selected experts per token
E = total logical experts in the replica
P = expert-parallel degree
R = T * k route assignments
```

Assume experts are partitioned evenly, so one EP rank owns `E/P` experts.
Then under balanced routing:

```text
mean routes per expert = R / E = T * k / E
mean routes per EP rank = R / P = T * k / P
experts per EP rank = E / P
```

Therefore increasing `P` while holding `T`, `k`, and `E` fixed does **not**
reduce the average tokens/routes received by each expert. It reduces the
number of experts and aggregate route work owned by each rank.

EP and DP are usually separate process-group dimensions. If world size is
fixed and raising `P` reduces DP degree, `T` per DP replica may increase; in
that case `T*k/E` can increase rather than decrease. Always state which of
global batch, microbatch, DP degree, `E`, `k`, capacity, and expert placement
is held fixed.

### 3.1 What EP degree can change

Under a deliberately simple uniform independent-route approximation, the
total routes landing on one EP rank are:

```text
X_rank ~ Binomial(R, 1/P)
E[X_rank] = R/P
Var[X_rank] = R * (1/P) * (1 - 1/P)
CV[X_rank] = sqrt((P - 1) / R)
```

At fixed `R`, larger `P` increases rank-level **relative** noise while reducing
the mean aggregate work per rank. Real MoE routing is not iid: top-k choices,
auxiliary load balancing, expert affinities, token correlation, sequence
packing, capacity/drop policy, and expert placement all change the variance
and tail. Per-expert load can also be correlated across the experts colocated
on one rank.

The Rubin-relevant questions are consequently:

- how many messages/ranges and bytes reach each rank;
- whether fewer local experts reduce weight reuse or improve residency;
- whether rank and wave tails worsen relative to shorter compute time;
- whether counters, ring capacity, and scheduling policy absorb the observed
  distribution;
- whether a larger NVLink domain encourages increasing `E`, `P`, or both.

Per-expert mean load decreases only when a quantity such as `E`, `T`, `k`,
capacity/drop policy, or replication/placement changes. Do not attribute that
change to `P` alone.

## 4. Stage-By-Stage Opportunity Map

| Current stage / contract | Rubin opportunity | Challenge / proof obligation |
|---|---|---|
| Dispatch pulls same-layout route payloads from different peer/token addresses | Inline TMA descriptor update may retain layout and cheaply override base pointer or stride | Verify which dynamic peer addresses/strides are legal, descriptor lifetime, setup amortization, and whether metadata rather than TMA remains limiting |
| Linear1/Linear2 repeatedly load expert weights with large K | Doubled K processing and 22 TB/s HBM4 can reduce K-loop and weight-streaming time | Small-M expert GEMM utilization, tile tails, scale metadata, and pipeline waits can dominate before peak is reached |
| Linear1 epilogue performs SwiGLU, top-k weighting, amax, and requantization | Faster matrix stages raise the value of fusing epilogue work and may motivate different producer/consumer partitioning | Activation, reduction, conversion, scale writes, and L2-ring publication become a larger relative fraction |
| Linear2 writes disjoint remote BF16 top-k slots | Counted writes could publish completion per token block/range | Define visibility, counter scope, expected counts, generation/ABA protection, counter hotspots, and failure recovery |
| Whole-phase NVLink barrier precedes local fixed slot-order combine | Fine-grained readiness could let a token block combine once all its slots arrive | Preserve fixed local slot order; readiness says all inputs arrived, not which floating-point order to reduce them |
| One NVLink domain supplies peer-visible transport | NVLink 6 at 3.6 TB/s/GPU and a 260 TB/s NVL72 fabric increases in-rack EP headroom | Fabric scales less than local FP8 peak in the folded figures; cross-node EP still needs DeepEP/RDMA/collective hierarchy |
| Wave/ring scheduling absorbs route imbalance | Faster stages and fine-grained dependencies can shorten bubbles or enable more dynamic work release | Fixed scheduling overhead, ring fragmentation, rank-level CV, and stragglers can become relatively more visible |

## 5. The Most Direct Opportunities

### 5.1 Same layout, different expert base pointer

Expert FFNs repeatedly use tensors with the same logical shape and layout but
different expert/peer base addresses. Rubin's inline TMA descriptor update is
therefore a close match to the current persistent scheduler:

```text
stable descriptor layout
  + next expert / route chooses base pointer or stride
  -> less descriptor management between otherwise similar transfers
```

This is a candidate, not a guaranteed win. If expert selection metadata,
remote latency, ring readiness, or small-M GEMM utilization dominates, cheaper
descriptor update will not move end-to-end throughput.

### 5.2 Replace phase completion with block readiness

The current combine contract is safe and simple:

```text
all remote route writes complete
  -> whole-phase NVLink barrier
  -> each token owner sums top-k slots in fixed order
```

Counted writes suggest a finer protocol:

```text
each remote writer publishes one completed range
  -> token block observes expected count/generation
  -> token block sums its already-disjoint slots in fixed slot order
```

This can overlap late write-back with early combine without changing the local
floating-point association. It must not collapse “arrival order” and “reduce
order” into one concept.

A correct design needs at least:

- one unambiguous expected count per readiness unit;
- release/acquire visibility from payload write to readiness observation;
- generation numbers for reused buffers and ring wraparound;
- a plan for duplicated/missing writes and kernel failure;
- counter granularity that avoids one hot global location;
- a fixed local slot order if deterministic combine is required.

### 5.3 Rebalance the fused pipeline

Using the folded public figures as direction only, Rubin versus GB300 is about
`3.5x` in FP8 peak, `2.75x` in HBM bandwidth, and `2x` in rack NVLink
bandwidth. Thus compute can improve faster than fabric, while HBM improves
faster than fabric but slower than FP8 compute.

The likely outcome is bottleneck migration:

```text
expert GEMM gets shorter
  -> dispatch/combine bytes, activation epilogue, counters, and tails occupy
     more of the critical path
```

This does not mean communication gets slower. It means the port must measure
the full fused timeline instead of extrapolating from Tensor Core peak.

## 6. Opportunities That Need Model Or System Co-Design

### 6.1 Three-bit LUT weights

The public 3-bit LUT matrix-B format is naturally relevant to inference expert
weights because expert FFNs are weight-heavy matrix operations. It is not a
transparent replacement for the current FP8/FP4 training path. A usable design
must define weight conversion/training recipe, scale ABI, accuracy, supported
MMA shapes, epilogue compatibility, and whether memory savings improve expert
residency enough to change scheduling.

### 6.2 Activation 2:4 compression

Rubin activation sparsity may help surrounding workloads, but public material
does not establish a drop-in MegaMoE training transformation. SwiGLU outputs,
router behavior, top-k weights, and expert gradients are model semantics.
Changing their sparsity requires an explicit quality/training contract rather
than treating a hardware format as lossless.

### 6.3 SHARP

SHARP in-network collective compute can accelerate compatible reductions and
collectives. Dynamic MoE token dispatch is a variable-count personalized
all-to-all, not automatically a reduction. Use SHARP only after identifying a
specific compatible collective in routing statistics, gradient reduction, or
another surrounding stage; do not cite it as direct proof of dispatch speedup.

## 7. Challenges That Remain

- **Small-M utilization:** more peak K throughput does not guarantee that a
  token-poor expert tile reaches it.
- **Rank/wave tails:** the slowest route group can still set step time even
  when mean traffic fits easily.
- **Relative fabric pressure:** local compute peak grows faster than folded
  rack NVLink bandwidth in the public numbers.
- **Counter protocol complexity:** fine readiness trades a simple phase
  barrier for more state, ordering, reuse, and recovery obligations.
- **Cross-node boundary:** NVL72 is a fast in-rack domain, not a proof that the
  symmetric-pointer kernel crosses RDMA domains.
- **Backward remains open:** dX, dW, router/auxiliary gradients, expert-gradient
  reductions, determinism, and optimizer interaction need a separate design.
- **Public architecture gaps:** shared memory, TMEM, exact instruction forms,
  occupancy, and latency details are not yet pinned well enough to choose a
  concrete Rubin kernel schedule.

## 8. Validation Plan

| Hypothesis | Required comparison |
|---|---|
| Inline descriptor update materially helps dispatch/weight loads | Same route/shape trace with descriptor update on/off; count setup instructions, issue stalls, and end-to-end wave time |
| Larger K processing helps expert GEMM | Sweep per-expert M, K, N, quantization, expert count, and route skew; report achieved throughput and tail latency, not only peak |
| Fine-grained combine beats whole-phase barrier | Compare barrier and counted-write designs over balanced/skewed traces; include counter overhead, early-combine overlap, and buffer reuse |
| Deterministic combine is preserved | Repeated bitwise tests with fixed routing/topology plus an audit that payload readiness is separate from fixed slot reduction order |
| NVLink 6 changes the best EP degree | Joint sweep of EP/DP, `E`, tokens per DP replica, topology, ring capacity, and cross-node hierarchy; report mean and tail rank load |
| HBM4 changes expert residency/streaming | Measure weight bytes, L2 hit rate, achieved HBM bandwidth, expert reuse distance, and capacity effects |

For route-distribution reports, always include:

```text
T, k, E, P, DP degree, global/micro batch, capacity/drop policy,
expert placement/replication, mean and percentile routes per expert,
mean and percentile routes/bytes per rank, and wave tail time
```

Without this envelope, “larger EP caused smaller experts” or “Rubin fixed
imbalance” is not an auditable conclusion.

## 9. Durable Conclusions

- Rubin's closest match to the current MegaMoE protocol is inline TMA
  descriptor update for same-layout/different-address expert data and counted
  writes for finer remote-output readiness.
- The current whole-phase combine barrier may become a token/block readiness
  protocol, but deterministic combine still needs a fixed reduction order.
- HBM4 and larger K processing can accelerate expert weight streaming/GEMM;
  activation, quantization, metadata, communication, and tails may then become
  more exposed.
- Larger EP degree alone does not reduce mean routes per expert. It changes
  per-rank ownership, communication/scheduling granularity, and distribution
  tails; per-expert mean is `T*k/E` under the stated envelope.
- NVLink 6 strengthens in-rack EP, while cross-node EP and backward remain
  separate protocol/design problems.
- Low-bit LUT weights, activation sparsity, and SHARP are promising only with
  a specific compatible numerical or collective contract; they are not
  automatic drop-in speedups for the current training semantics.
