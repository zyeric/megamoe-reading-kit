# DeepSeek-V4 / DeepGEMM MegaMoE Combine Notes

Status: first-pass Linear2 epilogue / combine notes

Context metadata:

- Topic: Linear2 epilogue remote write-back and final local combine.
- Layer tags: `stage-combine`, `runtime-protocol`, `distributed-ep`,
  `numerics`.
- Owns: source metadata consumption, remote BF16 route-output write-back,
  pre-combine barrier, top-k-slot summation, and combine determinism
  boundaries.
- Does not own: Linear2 GEMM body, dispatch source-rank interleaving, or
  symmetric-memory backend internals.
- Agent entry: read when the task asks "where does Linear2 output go?" or
  "why does final combine not use atomic add?".

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - `combine_token_buffer`, phase barriers, and source metadata lifetime.
- [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md) -
  route metadata, `TokenSrcMetadata`, and pool-token order.
- [`deepseek-v4-megamoe-gemm.md`](deepseek-v4-megamoe-gemm.md) - shared
  Linear1 / Linear2 GEMM body and Linear2 FP32 TMEM accumulator production.
- [`deepseek-v4-megamoe-activation.md`](deepseek-v4-megamoe-activation.md) -
  top-k weight is applied before Linear2 in the public FP8/FP4 path.
- [`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-cuda-symmetric-memory)
  - peer-addressability and symmetric-memory caveats.

Scope:

- Track the public DeepGEMM SM100 FP8/FP4 MegaMoE forward combine path.
- Cover Linear2 epilogue remote write-back and final top-k-slot summation.
- Keep Linear2 GEMM body in the GEMM note.
- Keep training backward and cross-node RDMA claims out of scope.

Current evidence level:

- Code-checked against public DeepGEMM `main` files:
  `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` and
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`.
- This note describes the public SM100 FP8/FP4 forward path. It should not be
  generalized to unpublished backward kernels or non-SM100 implementations
  without new evidence.

## One-Line Model

Combine has two stages inside the fused kernel:

```text
Linear2 epilogue:
  FP32 TMEM C tile
  -> BF16 route output
  -> remote token-owner combine_token_buffer[topk_slot, token]

Final combine reduction:
  wait cross-rank barrier
  -> for each local token, load valid top-k slot outputs
  -> sum preweighted BF16 route outputs in FP32 registers
  -> cast to BF16 and write y[token]
```

The key semantic point:

```text
top-k weights are already applied in the Linear1 activation epilogue
```

So the final combine step is a sum over preweighted route outputs. It does not
read or multiply `topk_weights` again.

## Code Anchors

DeepGEMM paths:

- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` - `TokenSrcMetadata`,
  workspace layout, and full-pool source metadata storage.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` - Linear2
  epilogue write-back, `combine_token_buffer`, pre-combine NVLink barrier, and
  final top-k-slot summation.

Important variables:

| Name | Meaning |
|---|---|
| `TokenSrcMetadata` | `{rank_idx, token_idx, topk_idx}` for one logical pool token |
| `pool_m_idx + m_idx_in_block` | full-pool token index used to recover source metadata |
| `combine_token_buffer` | per-topk-slot BF16 route-output buffer inside the symmetric buffer |
| `dst_rank_idx` | original token-owner rank |
| `dst_token_idx` | original token index on the token-owner rank |
| `dst_topk_idx` | top-k slot index for that route |
| `kBeforeCombineReduceBarrierTag` | cross-rank barrier before final local combine reduction |
| `kNumChunks` | split hidden dimension into one or two chunks for combine buffering |

## Source Metadata Comes From Dispatch

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

This metadata is stored in the full expert-token pool, not only in a reusable
ring slot:

```text
pool token -> original rank, original token, original top-k slot
```

That is necessary because by the time Linear2 writes route outputs back, the
physical L1/L2 ring slot may already be reused for another generation. Combine
needs the logical source identity, not the temporary ring slot identity.

## Linear2 Epilogue: Remote Route Write-Back

Linear2 GEMM ends with a complete FP32 C tile in TMEM. The Linear2 epilogue:

1. loads accumulator values from TMEM;
2. casts them to BF16;
3. stages BF16 route-output rows in shared memory;
4. uses `TokenSrcMetadata` to find the original token-owner rank and top-k slot;
5. writes the route output to the remote combine buffer.

The data path is:

```text
TMEM FP32 accumulator
  -> BF16 in shared_storage.smem_d.l2
  -> sym_buffer.map(remote combine_token_buffer slot)
```

The code reads source metadata per valid row:

```cpp
src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);

dst_rank_idx  = src_metadata.rank_idx;
dst_token_idx = src_metadata.token_idx;
dst_topk_idx  = src_metadata.topk_idx;
```

Then it writes to the token-owner rank's symmetric buffer:

```cpp
dst_token = combine_token_buffer
    .get_rank_buffer(dst_topk_idx)
    .get_data_buffer(dst_token_idx);

*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
```

`combine_token_buffer` is named as a `layout::Buffer` with `kNumTopk` slots:

```cpp
layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, ...)
```

Here the buffer's "rank" dimension is used as the top-k slot dimension. The
actual remote GPU rank is selected by `sym_buffer.map(..., dst_rank_idx)`.

Conceptually, this is a push from the expert-owner rank to the token-owner
rank:

```text
rank that computed Linear2 route output
  pushes BF16 route output
  into original token-owner rank's combine_token_buffer[topk_slot, token]
```

It is not implemented as a final-combine pull from remote ranks. The remote
store happens in the Linear2 epilogue, before the pre-combine barrier. After the
barrier, each rank reduces its own local `combine_token_buffer` slots for the
tokens it owns.

## Why No Atomic Add Is Needed During Write-Back

Each routed token-topk pair has a unique destination:

```text
(dst_rank_idx, dst_token_idx, dst_topk_idx)
```

So the Linear2 epilogue can write a full BF16 route output into a unique
combine slot. It does not need to atomically add into `y[token]`.

This is an important determinism choice:

```text
many remote writers:
  write disjoint top-k slots

one final local reducer:
  sums those slots in a fixed slot order
```

The price is an intermediate combine buffer of shape conceptually:

```text
[num_topk, num_max_tokens_per_rank, hidden]
```

inside the symmetric buffer.

## Pre-Combine Barrier

After the scheduler finishes its Linear1 / Linear2 block loop, the kernel
performs an NVLink-domain barrier before final reduction:

```cpp
scheduler.for_each_block([&](...) {
    if (block_phase == BlockPhase::Linear1) {
        // Linear1 activation epilogue writes l2_acts / l2_acts_sf.
    } else {
        // Linear2 epilogue writes remote combine_token_buffer slots.
        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
    }
});

comm::nvlink_barrier<..., kBeforeCombineReduceBarrierTag>(...);

for (uint32_t token_idx = ...; token_idx < num_tokens; ...) {
    // Final local combine reduction.
}
```

In the source, the barrier is:

```cpp
comm::nvlink_barrier<... kBeforeCombineReduceBarrierTag>(...);
```

The source comment describes this as:

```text
grid sync + cross-rank signal + grid sync
```

This is a phase-level barrier. It is after all scheduled L1/L2 blocks have run
and after all Linear2 route-output write-backs have been issued. It is not
per-token, not per-expert, and not per-wave.

The purpose is not to make the result deterministic by sorting. The purpose is
remote-write completion / visibility:

```text
all remote combine_token_buffer writes are visible
  before
token-owner ranks read their local combine slots
```

After this barrier, epilogue warps perform final combine. Dispatch warps can
also proceed to workspace cleanup through a separate synchronization point;
the source comment notes that cleanup is overlapped with the combine reduction
epilogue.

The consequence is that final combine does not start as soon as one token's
top-k route outputs are ready. The public path waits for the whole Linear2
write-back phase, then performs final local reductions. The overlap here is
between workspace cleanup and the combine reduction after the barrier, not a
fine-grained "token ready -> immediately combine" pipeline.

## Final Combine Reduction

Final combine is local to the token-owner rank. For each local token, one warp
handles chunks of the hidden dimension:

```text
for token_idx assigned to this warp:
  read valid top-k slots from input_topk_idx[token_idx]
  for each hidden chunk:
    TMA-load combine_token_buffer[slot, token_idx, chunk]
    accumulate BF16 slot values in FP32 registers
    cast reduced result to BF16
    TMA-store y[token_idx, chunk]
```

The code builds a mask from valid top-k slots:

```cpp
stored_topk_slot_idx =
    lane_idx < kNumTopk ? input_topk_idx[token_idx, lane_idx] : -1;

total_mask = ballot(stored_topk_slot_idx >= 0);
```

The source variable name is slightly misleading. `input_topk_idx[token, lane]`
stores the selected expert ID, or `-1` for a masked route. In this combine
loop, the value is only used to decide whether the top-k slot is valid. The
actual combine-buffer slot index is the lane / bit position in the top-k array.

Then it iterates set bits in increasing slot-index order:

```cpp
slot_idx = __ffs(mask) - 1;
mask ^= 1 << slot_idx;
```

For each selected slot, it TMA-loads the BF16 route output into a per-warp
shared-memory load buffer and accumulates into FP32 registers:

```cpp
ptx::tma_load_1d(combine_load_buffer[i], src_ptr, ...);

ptx::accumulate(reduced[...], bf16_values[l]);
```

Finally, it rounds to BF16 and TMA-stores the local output tensor:

```cpp
casted_bf16[l] = __float22bfloat162_rn(reduced[...]);
ptx::tma_store_1d(y + token_offset, combine_store_buffer, kNumChunkBytes);
```

Again, no `topk_weights` are read here. The combine reduction is:

```text
y[token] = sum over valid top-k slots:
  preweighted_route_output[token, slot]
```

## Chunking And Double-Buffered Loads

The combine reducer may split the hidden dimension into one or two chunks:

```cpp
kNumChunks = condition ? 1 : 2;
kNumChunkBytes = kHidden * sizeof(bfloat16) / kNumChunks;
```

The condition is constrained by shared-memory and register budget:

```text
2 load buffers + 1 store buffer
per epilogue warp
```

Within a chunk, it uses two load stages:

```text
load top-k slot s+1
  while
accumulating top-k slot s
```

This is a small local pipeline:

```text
combine_token_buffer global memory
  -> TMA load into shared-memory load buffer
  -> FP32 register accumulation
  -> BF16 shared-memory store buffer
  -> TMA store to y
```

## Determinism Notes

Forward combine determinism in the public path comes from:

- one writer per `(token, topk_slot)` route-output buffer;
- a barrier before reading combine slots;
- fixed slot traversal order from low to high set bit in the top-k mask;
- FP32 register accumulation followed by one BF16 rounding at output store.

Potential caveats:

- The slot order is deterministic, but it is slot order, not sorted expert ID
  order. `tests/test_mega_moe.py` uses `torch.topk(..., sorted=False)`, so
  slot assignment is whatever the router/top-k output contract produced.
- This note only covers forward combine. Backward may have different reduction
  patterns and atomic / accumulation risks.
- Cross-rank visibility is inferred from the public kernel's NVLink barrier
  and symmetric-memory write path; do not generalize this to cross-node RDMA
  without separate evidence.

## Boundary To Future Work

This combine note closes the public forward path:

```text
dispatch
  -> Linear1 GEMM
  -> activation epilogue
  -> Linear2 GEMM
  -> Linear2 write-back
  -> final combine reduction
```

Remaining work should be treated separately:

- compare this fused combine path against DeepEP dispatch/combine baselines;
- inspect whether any public issue / PR exposes backward or training kernels;
- build a forward determinism checklist across dispatch, GEMM, activation, and
  combine;
- only then move to backward / training claims.
