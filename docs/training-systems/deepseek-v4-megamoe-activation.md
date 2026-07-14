# DeepSeek-V4 / DeepGEMM MegaMoE Activation Epilogue Notes

Status: first-pass activation / SwiGLU epilogue notes

Context metadata:

- Topic: Linear1 epilogue, SwiGLU, top-k weighting, and L2 ring publish.
- Layer tags: `stage-activation`, `numerics`, `runtime-protocol`.
- Owns: gate/up TMEM reads, BF16-rounded activation path, top-k weight
  application, amax/scale computation, FP8 requantization, and
  `l2_full_count`.
- Does not own: Linear1 GEMM body, generic ring-buffer protocol, or final
  combine.
- Agent entry: read after GEMM/quantization when the task asks "what happens
  between Linear1 and Linear2?".

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - L2 ring slots and producer-consumer counters.
- [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md) -
  token pull and `l1_topk_weights_buffer`.
- [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md) -
  wave / pool-block / ring-block scheduling.
- [`deepseek-v4-megamoe-quantization.md`](deepseek-v4-megamoe-quantization.md)
  - FP8/FP4 payloads, UE8M0 scale factors, and L1 gate/up weight interleave.
- [`gpu-hardware-notes/notes/cuda-kernel-patterns.md`](https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md)
  - epilogue, TMA, UMMA, TMEM, ring-buffer, and producer-consumer patterns.

Scope:

- Track the public DeepGEMM SM100 FP8/FP4 MegaMoE forward activation path.
- Focus on Linear1 epilogue: gate/up interpretation, clamp, SiLU, top-k
  weight application, amax / scale, FP8 requantization, and `l2_full_count`.
- Keep GEMM determinism and Linear2 / combine overlap as separate notes.

Current evidence level:

- Code-checked against public DeepGEMM `main` files:
  `tests/test_mega_moe.py`,
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`,
  `deep_gemm/include/deep_gemm/common/math.cuh`, and
  `third-party/tilelang_ops/swiglu.py`.
- This note describes the public SM100 FP8/FP4 forward path. It should not be
  generalized to unpublished training backward kernels or non-SM100 paths
  without new evidence.

## One-Line Model

The fused MegaMoE activation is not a standalone TileLang kernel. It is the
Linear1 epilogue inside the SM100 CUDA/CuTe MegaMoE kernel:

```text
Linear1 UMMA FP32 accumulator in TMEM
  -> epilogue reads gate/up pairs from TMEM
  -> BF16 rounding + clamp + SiLU(gate) * up * topk_weight
  -> amax over 32-channel groups
  -> UE8M0 scale + FP8 E4M3 l2_acts
  -> TMA store to L2 ring buffer
  -> publish l2_full_count for Linear2
```

This is why activation belongs to the epilogue layer in the lowering map:

```text
math:
  hidden = silu(gate) * up

execution rewrite:
  hidden is never materialized as a full BF16 tensor
  it is produced from TMEM and immediately requantized into l2_acts ring slots
```

## Fused Path Versus TileLang Baseline

`tests/test_mega_moe.py` uses TileLang only in the non-overlapped baseline:

```text
DeepEP dispatch
  -> grouped L1 GEMM
  -> tilelang_ops.swiglu_apply_weight_to_fp8
  -> grouped L2 GEMM
  -> DeepEP combine
```

The baseline code makes this explicit:

```python
l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2),
                   dtype=torch.bfloat16, device="cuda")
gemm_fn(recv_x, l1_weights, l1_y, ...)

swiglu_result = tilelang_ops.swiglu_apply_weight_to_fp8(
    x=l1_y,
    topk_weights=recv_topk_weights,
    num_per_channels=32,
    use_col_major_scales=True,
    ue8m0_scale=True,
    output_bf16=False,
)

gemm_fn(l1_y, l2_weights, l2_y, ...)
ep_buffer.combine(l2_y, handle=handle)
```

The TileLang op is a Python DSL kernel with `@tilelang.jit`, `T.prim_func`,
and `T.Kernel(...)`. It is useful as a readable reference for the formula and
scale-factor shape, but it is not the fused MegaMoE implementation.

The fused path does the corresponding work inside
`sm100_fp8_fp4_mega_moe.cuh`, in the `BlockPhase::Linear1` epilogue branch.

## Code Anchors

DeepGEMM paths:

- `tests/test_mega_moe.py` - baseline sequence and correctness comparison.
- `third-party/tilelang_ops/swiglu.py` - baseline SwiGLU + FP8 cast kernel.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` - fused
  MegaMoE kernel, Linear1 epilogue, TMA stores, and producer-consumer counters.
- `deep_gemm/include/deep_gemm/common/math.cuh` - `get_e4m3_sf_and_sf_inv`.

Important variables in the fused kernel:

| Name | Meaning |
|---|---|
| `L1_SHAPE_N = kIntermediateHidden * 2` | Linear1 produces gate and up channels |
| `L1_OUT_BLOCK_N = BLOCK_N / 2` | post-SwiGLU output width for one L1 N block |
| `l1_topk_weights_buffer` | per-routed-token top-k weight copied during dispatch pull |
| `kNumEpilogueStages = 2` | number of TMEM accumulator stages between UMMA and epilogue |
| `accum_stage_idx` | TMEM accumulator stage index, selected by `current_iter_idx % kNumEpilogueStages` |
| `accum_phase` | phase bit used when a TMEM stage is reused across generations |
| `shared_storage.tmem_full_barriers` | UMMA has produced a TMEM accumulator stage |
| `shared_storage.tmem_empty_barriers` | epilogue has consumed a TMEM accumulator stage |
| `shared_storage.smem_d.l1` | shared-memory staging for FP8 post-SwiGLU output before TMA store |
| `l2_token_buffer` / `l2_sf_buffer` | global-memory L2 ring payload and scale-factor buffers |
| `l2_full_count` | Linear1 has produced enough L2 input chunks for a ring-block generation |

## Question 1: How Gate / Up Are Read From TMEM

Linear1 computes:

```text
[gate, up] = x @ W1^T
```

The public FP8/FP4 kernel uses block-scaled UMMA. The accumulator is FP32 and
lives in SM100 TMEM until the epilogue consumes it.

The epilogue first waits for UMMA:

```cpp
shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
ptx::tcgen05_after_thread_sync();
```

Then the Linear1 branch reads TMEM accumulator values:

```cpp
uint32_t tmem_addr =
    accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;

cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, ...);
cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, ...);
```

Those raw TMEM values are interpreted as `float2` pairs:

```cpp
auto fp32_values = reinterpret_cast<float2*>(raw_values);
bf16_gate = bf16_round(fp32_values[k * 2 + 0]);
bf16_up   = bf16_round(fp32_values[k * 2 + 1]);
```

This only works efficiently because the L1 gate/up weights were transformed
before the kernel. The source comment calls this "granularity 8 interleaved
weights": DeepGEMM interleaves L1 gate/up output channels in small groups, so
a TMEM epilogue load sees adjacent gate/up pairs instead of needing to gather
from two distant halves of the Linear1 output.

Conceptually:

```text
unfriendly logical layout:
  [gate_0, gate_1, ..., gate_I-1, up_0, up_1, ..., up_I-1]

epilogue-friendly physical layout:
  [gate group, up group, gate group, up group, ...]
```

For each `BLOCK_N = 128` Linear1 output block, the epilogue collapses gate/up
into `L1_OUT_BLOCK_N = 64` post-activation columns. Those 64 columns become a
slice of `l2_acts`, the input K dimension for Linear2.

## Question 1.5: Is Linear1 -> Activation Also A Pipeline?

Yes. Linear1 UMMA and activation epilogue have their own producer-consumer
pipeline. This is lower-level than the global ring-buffer pipeline:

```text
global ring-buffer pipeline:
  dispatch -> Linear1 -> activation epilogue -> Linear2 -> combine

TMEM accumulator pipeline:
  Linear1 UMMA -> TMEM accumulator stage -> activation epilogue
```

The producer-consumer object is a TMEM accumulator stage, not `l1_acts` or
`l2_acts` global memory. The public SM100 kernel uses two epilogue stages:

```cpp
constexpr uint32_t kNumEpilogueStages = 2;
accum_stage_idx = current_iter_idx % kNumEpilogueStages;
accum_phase = (current_iter_idx++ / kNumEpilogueStages) & 1;
```

The UMMA issue warp is the producer. Before writing a TMEM stage, it waits for
that stage to be empty:

```cpp
shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
```

Then it runs the K reduction for the selected `(phase, expert, m_block,
n_block)` tile. The important detail is that epilogue does not consume partial
K results. The UMMA loop issues all K-block UMMAs into the same accumulator
stage and only signals `tmem_full` on the last K block:

```cpp
for (k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
    // wait TMA-loaded A/B/SF tiles
    shared_storage.full_barriers[stage_idx].wait(phase);

    // issue one or more UMMA instructions; accumulate into accum_stage_idx
    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(...);

    // only the final K block publishes the complete C tile
    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
}
```

The epilogue warps are the consumer. They traverse the same scheduler order and
wait for the matching TMEM stage to become full:

```cpp
shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
ptx::tcgen05_after_thread_sync();
```

After they have read the complete accumulator tile from TMEM, they release that
stage so UMMA can reuse it for a later logical block:

```cpp
SM100_TMEM_LOAD_16dp256b1x::copy(...);

// on the last atom of the tile
ptx::tcgen05_before_thread_sync();
shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
```

The shape of the overlap is:

```text
time 0:
  UMMA writes block 0 into TMEM stage 0

time 1:
  epilogue consumes block 0 from stage 0
  UMMA writes block 1 into stage 1

time 2:
  epilogue consumes block 1 from stage 1
  UMMA writes block 2 into stage 0
```

The phase bit disambiguates different generations when a physical stage is
reused. The same pattern appears again at a larger scale for `l2_acts`, but the
storage and synchronization are different:

| Pipeline | Producer output | Storage | Consumer wait |
|---|---|---|---|
| UMMA -> activation epilogue | complete Linear1 C tile | TMEM accumulator stage | `tmem_full_barriers` |
| activation epilogue -> Linear2 | FP8 `l2_acts` + UE8M0 `l2_acts_sf` | global-memory ring slot | `l2_full_count` |

Execution-resource note:
[`gpu-hardware-notes/notes/cuda-kernel-patterns.md`](https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md)
and [`gpu-hardware-notes/notes/gpu-execution-model.md`](https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md)
track the hardware background. In this epilogue, UMMA / Tensor Cores produce
the FP32 accumulator tile. The epilogue then uses SM100 TMEM load instructions
to move accumulator values into registers. The following BF16 rounding, clamp,
SiLU / `exp`, top-k multiply, amax, FP8 conversion, shared-memory stores, and
TMA store issue are scalar-vector / SFU / memory-path work, not another Tensor
Core GEMM.

## Question 2: What Is The Precision Path

The fused activation path is:

```text
FP32 accumulator in TMEM
  -> round gate/up to BF16
  -> optional clamp
  -> convert BF16 gate/up back to float
  -> SiLU(gate) * up
  -> multiply top-k weight
  -> FP32 activation value before requantization
```

The code does the BF16 rounding before the nonlinear math:

```cpp
auto bf16_gate = __float22bfloat162_rn(fp32_gate);
auto bf16_up   = __float22bfloat162_rn(fp32_up);
```

This BF16 cast should be read as a numeric contract boundary, not as a TMEM
capacity requirement. The accumulator is FP32 because the Linear1 K reduction
benefits from higher-precision accumulation. But the public baseline materializes
the Linear1 output as BF16 before applying SwiGLU:

```python
l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2),
                   dtype=torch.bfloat16, device="cuda")
```

If the fused epilogue used the raw FP32 TMEM accumulator directly for SiLU, it
would compute a different function from the baseline:

```text
silu(round_bf16(gate)) * round_bf16(up)
  != silu(fp32_gate) * fp32_up
```

So the intended precision boundary is:

```text
Linear1 GEMM:
  FP32 accumulator in TMEM for reduction quality

Activation input:
  BF16-rounded gate/up to match the mixed-precision forward contract

Linear2 input:
  FP8 E4M3 payload + UE8M0 scale factors for bandwidth and SM100 UMMA
```

If activation clamp is enabled, the exact clamp is asymmetric for gate and
symmetric for up:

```text
gate = min(gate, clamp)
up   = min(max(up, -clamp), clamp)
```

Then the epilogue computes:

```text
gate = gate / (1 + exp(-gate))
activation = gate * up * topk_weight
```

`kFastMath` changes the implementation of `exp` / reciprocal:

```text
kFastMath = true:
  __expf + approximate reciprocal

kFastMath = false:
  expf + ordinary division
```

The top-k weight is not applied in the final combine step in the current public
FP8/FP4 MegaMoE contract. Dispatch pull copies one route weight into
`l1_topk_weights_buffer`; Linear1 epilogue reads it and multiplies the
activation before Linear2:

```cpp
stored_cached_weight = *l1_topk_weights_buffer
    .get_data_buffer(ring_m_idx + ...)
    .get_base_ptr<float>();

activation_values[i][k] = silu(gate) * up * weights;
```

This is mathematically valid because Linear2 is linear:

```text
topk_weight * W2(hidden) == W2(topk_weight * hidden)
```

So the per-route Linear2 output is already weighted when it reaches the combine
buffer. Combine can reconstruct each token by summing the routed outputs.
The combine code reads `topk_idx` to find valid slots, TMA-loads those BF16
slot buffers, and accumulates them in float registers; it does not read
`topk_weights` again.

## Question 3: How Amax / Scale / FP8 Requant Maps To `l2_acts_sf`

The Linear1 epilogue immediately requantizes the post-SwiGLU activation for
Linear2. It does not write a full BF16 intermediate tensor.

The baseline TileLang op exposes the high-level scale rule:

```text
num_per_channels = 32
amax = max(abs(y[token, 32g : 32g + 32]))
sf = ceil_pow2(amax / 448)
y_fp8 = fp8_e4m3(y / sf)
```

The fused CUDA epilogue implements the same 32-channel idea with warp and
warp-pair reductions.

Local activation values are first reduced inside a thread:

```cpp
thread_local_amax = max(abs(activation_values...));
```

Then the code reduces across lanes and across a pair of epilogue warps:

```cpp
amax_values[i] = math::warp_reduce<4, true>(
    thread_local_amax,
    math::ReduceMax<float>());

shared_storage.amax_reduction[epilogue_warp_idx][...] = amax_values[i];
wp_amax = shared_storage.amax_reduction[epilogue_warp_idx ^ 1][...];
amax_values[i] = max(amax_values[i], wp_amax);
```

After amax is known, the epilogue computes UE8M0 scale and inverse scale:

```cpp
math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);
```

`get_e4m3_sf_and_sf_inv` uses the FP8 E4M3 max finite value `448`:

```text
sf     = pow2(ceil(log2(amax / 448)))
sf_inv = 1 / sf
```

The activation payload is scaled and converted to FP8 E4M3:

```cpp
upper = activation_values[i][0] * sf_inv;
lower = activation_values[i][1] * sf_inv;
fp8x4_values = __nv_fp8x4_e4m3(...);
```

Those FP8 values are first stored into shared memory:

```cpp
ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
```

Then one elected warp issues a TMA store from shared memory into the global
`l2_acts` ring buffer:

```cpp
cute::SM90_TMA_STORE_2D::copy(
    &tensor_map_l1_output,
    shared_storage.smem_d.l1[...],
    out_n_idx,
    ring_m_idx + ...);
```

The corresponding scale bytes go to `l2_sf_buffer`:

```cpp
sf_base_ptr[sf_addr] = exponent_byte(sf.x);
sf_base_ptr[sf_addr + 4 * sizeof(uint32_t)] = exponent_byte(sf.y);
```

Important layout points:

- `l2_acts` stores FP8 payloads in global memory ring slots.
- `l2_acts_sf` / `l2_sf_buffer` stores UE8M0 scale bytes in the transposed /
  M-major layout expected by the later UMMA SFA load.
- Four UE8M0 scale bytes pack into one `uint32_t` word.
- `k_idx = n_block_idx * 2 + warp_idx_in_wg / 2` reflects that one original
  `BLOCK_N = 128` Linear1 block becomes two 32-channel scale groups after
  gate/up collapse.

This is the same boundary as the quantization note:

```text
Linear1 accumulator:
  FP32 in TMEM

Linear2 input:
  FP8 E4M3 l2_acts + UE8M0 l2_acts_sf
```

## Question 4: How Linear1 Publishes To Linear2

The `l2` ring buffer name means "Linear2 input", not "L2 cache". In this public
MegaMoE path, `l2_acts` and `l2_acts_sf` are global-memory ring-buffer views
inside the symmetric buffer. Physically they are backed by local GPU memory
and can be cached by L2, but L2 cache is not the program object that carries
correctness.

The useful distinction is:

```text
program-visible buffer:
  l2_acts / l2_acts_sf global-memory ring slots
  addressed by tensor maps and protected by l2_full_count / l2_empty_count

hardware cache behavior:
  Linear1 TMA stores may leave recently written lines in L2
  Linear2 TMA loads may hit L2 if the data has not been evicted
  correctness cannot assume that residency
```

So the activation epilogue writes to a stable global-memory address. L2 cache
may make the later Linear2 load cheaper, but it is not where the kernel
allocates or synchronizes the ring buffer.

The Linear2 input ring buffer is reusable. Before Linear1 writes a physical
ring slot, it waits until Linear2 has released the previous generation of that
slot:

```cpp
while (ld_acquire(l2_empty_count[ring_block_idx]) != expected_previous_blocks) {
    // spin
}
```

After the epilogue has:

1. consumed the TMEM accumulator;
2. computed SwiGLU and top-k weighting;
3. stored FP8 payloads into shared memory;
4. TMA-stored those payloads into `l2_acts`;
5. written corresponding UE8M0 bytes into `l2_sf_buffer`;

it waits for the TMA stores to complete and then publishes the block:

```cpp
ptx::tma_store_wait<0>();
ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

ptx::red_add_rel(workspace.get_l2_full_count_ptr(ring_block_idx), 1u);
ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
```

`l2_full_count` is counted per produced Linear1 N-block. Linear2 waits for the
whole intermediate activation for that ring-block generation:

```cpp
num_expected_blocks =
    (L2_SHAPE_K / BLOCK_N) * 2 * (pool_block_idx / kNumRingBlocks + 1);

while (ld_acquire(l2_full_count[ring_block_idx]) != num_expected_blocks) {
    // spin
}
```

The factor `* 2` comes from the gate/up collapse:

```text
Linear1 N dimension:
  2 * intermediate_hidden

one Linear1 BLOCK_N = 128 block:
  produces BLOCK_N / 2 = 64 post-SwiGLU columns

number of produced L2 K chunks:
  (intermediate_hidden / 128) * 2
```

This means Linear2 does not start from a single 32-channel scale group. It
starts when the ring block has enough produced L2 input chunks for the whole
intermediate K dimension. The pipeline overlap is therefore mostly across
pool blocks / ring-block generations, not within one token block's individual
activation scale group.

## Mental Execution Example

Suppose:

```text
BLOCK_N = 128
intermediate_hidden = 256
```

Linear1 has logical output width:

```text
L1_SHAPE_N = 2 * 256 = 512
```

So Linear1 has four N blocks:

```text
L1 n_block 0: gate/up physical block -> l2_acts columns 0..63
L1 n_block 1: gate/up physical block -> l2_acts columns 64..127
L1 n_block 2: gate/up physical block -> l2_acts columns 128..191
L1 n_block 3: gate/up physical block -> l2_acts columns 192..255
```

Each block increments `l2_full_count` once after its FP8 payload and scale
bytes are visible. Linear2 waits until `l2_full_count` has advanced by four
for that ring-block generation, then it can run its K-tiled GEMM over the
complete `intermediate_hidden = 256` input.

## Current Open Questions

- Need generated PTX / SASS or NVIDIA documentation to fully decode the TMEM
  address bit `0x00100000` and the exact lane-to-channel mapping of
  `SM100_TMEM_LOAD_16dp256b1x`.
- Need GPU profiling to estimate whether this epilogue is compute-bound,
  shared-memory / TMA-store-bound, or mostly hidden by the GEMM pipeline.
- Need a separate backward-path source before making claims about training
  activation gradients, `dW`, or `dX` kernels.
