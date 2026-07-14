# DeepSeek-V4 / DeepGEMM MegaMoE Quantization Notes

Status: first-pass scale-factor path notes

Context metadata:

- Topic: FP8/FP4 payload and scale-factor path for public MegaMoE forward.
- Layer tags: `numerics`, `stage-dispatch`, `stage-gemm`,
  `stage-activation`.
- Owns: `x` / `x_sf`, FP8 E4M3 activation payloads, FP4 E2M1 weights,
  UE8M0 scale factors, SFA/SFB layout, and kernel ABI dtype boundaries.
- Does not own: routing policy, generic ring-buffer protocol, or final combine
  reduction mechanics.
- Agent entry: read when a question mentions FP8, FP4, BF16, scale factors,
  SFA/SFB, UE8M0, or where quantization happens.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - symmetric buffer, pool/ring slots, and counters.
- [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md) -
  token/SF pull into local L1 ring slots.
- [`deepseek-v4-megamoe-scheduling.md`](deepseek-v4-megamoe-scheduling.md) -
  wave / pool-block / ring-block scheduling.
- [`gpu-hardware-notes/notes/cuda-kernel-patterns.md`](https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md)
  - TMA, UMMA, TMEM, persistent kernels, and GEMM pipeline stages.

External reference:

- DeepSeek-V3 Technical Report, section 3.3.2 and appendix B.2:
  <https://arxiv.org/html/2412.19437>
- DeepSeek-V4 Technical Report, section 3.1 and section 5.3:
  [`../sources/2606.19348v1-deepseek-v4.pdf`](../sources/2606.19348v1-deepseek-v4.pdf)

Scope:

- Track the public DeepGEMM SM100 FP8/FP4 MegaMoE forward quantization path.
- Focus on `x_sf`, `l1_acts_sf`, `l2_acts_sf`, weight scale factors, and the
  scale-factor layout expected by TMA / UMMA.
- Keep the full SwiGLU epilogue, GEMM determinism, and combine overlap as
  separate follow-up notes.

Current evidence level:

- Code-checked against public DeepGEMM `main` files:
  `deep_gemm/mega/__init__.py`, `csrc/apis/mega.hpp`,
  `tests/test_mega_moe.py`,
  `deep_gemm/utils/math.py`,
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`, and
  `deep_gemm/include/deep_gemm/common/math.cuh`.
- This note describes the public SM100 FP8/FP4 forward path. It should not be
  generalized to unpublished training kernels or non-SM100 paths without new
  evidence.

## One-Line Model

In the FP8/FP4 MegaMoE path, token / activation values and scale factors move
as paired streams:

```text
BF16 input
  -> outside-kernel cast to per-token FP8 E4M3 input + UE8M0 scale factors
  -> MegaMoE kernel ABI: x FP8 payload + x_sf packed scale factors
  -> inside-kernel dispatch pull into L1 ring buffer
  -> Linear1 UMMA consumes FP8 activation + FP4 weight + SFA/SFB
  -> Linear1 epilogue computes SwiGLU/top-k-weighted FP32 values
  -> epilogue quantizes those values back to FP8 and writes L2 scale factors
  -> Linear2 UMMA consumes FP8 activation + FP4 weight + SFA/SFB
  -> Linear2 epilogue writes BF16 output to combine buffer
```

The important separation is:

```text
input / weight quantization happens before the MegaMoE kernel
intermediate activation quantization happens inside the Linear1 epilogue
final Linear2 output is BF16, not re-quantized for another MegaMoE stage
```

So "MegaMoE input" needs a layer qualifier:

```text
model / math layer:
  MoE receives BF16 hidden states from the previous model block

public DeepGEMM fp8_fp4_mega_moe kernel ABI:
  the caller must already provide x as FP8 E4M3 payload
  and x_sf as packed UE8M0 scale factors
```

## Code Anchors

DeepGEMM paths:

- `tests/test_mega_moe.py` - creates BF16 random inputs / weights, casts input
  tokens to FP8 and weights to FP4, copies `x` and `x_sf` into `SymmBuffer`,
  and compares fused output to the legacy baseline.
- `deep_gemm/mega/__init__.py` - defines `SymmBuffer`, slices the symmetric
  buffer into tensor views, and transforms weights / weight scale factors for
  MegaMoE.
- `deep_gemm/utils/math.py` - implements the public Python reference helpers
  `per_token_cast_to_fp8`, `per_token_cast_to_fp4`, UE8M0 rounding, and scale
  packing.
- `csrc/apis/mega.hpp` - computes the raw symmetric-buffer layout, creates
  tensor views and strides, validates FP8/FP4 scale-factor layouts, and
  dispatches to the SM100 kernel.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` - pulls
  input scale factors, TMA-loads activation and weight scale factors, feeds
  scale factors into UMMA, and stores Linear1 output scale factors.
- `deep_gemm/include/deep_gemm/common/math.cuh` - implements
  `get_e4m3_sf_and_sf_inv`.

## Tensor And Scale-Factor Views

From `csrc/apis/mega.hpp`, the FP8/FP4 path creates these public tensor views
inside the symmetric buffer:

| View | Shape | Dtype | Layout / role |
|---|---:|---|---|
| `x` | `[num_max_tokens_per_rank, hidden]` | FP8 E4M3 | local input token payload |
| `x_sf` | `[num_max_tokens_per_rank, hidden / 128]` | `torch.int` | packed input scale factors; K-major view |
| `l1_acts` | `[num_ring_tokens, hidden]` | FP8 E4M3 | Linear1 input ring payload |
| `l1_acts_sf` | `[num_sf_ring_tokens, hidden / 128]` with stride `{1, num_sf_ring_tokens}` | `torch.int` | Linear1 input scale factors; M-major / UTCCP-friendly view |
| `l2_acts` | `[num_ring_tokens, intermediate_hidden]` | FP8 E4M3 | Linear2 input ring payload |
| `l2_acts_sf` | `[num_sf_ring_tokens, intermediate_hidden / 128]` with stride `{1, num_sf_ring_tokens}` | `torch.int` | Linear2 input scale factors; M-major / UTCCP-friendly view |

The API code comments make the layout distinction explicit:

```cpp
// `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
```

The `_sf` suffix means "scale factor", but these tensors are not ordinary
float scale tensors in the hot path. For the FP8/FP4 path they are stored as
packed UE8M0-style scale metadata in `torch.int` / `uint32_t` views, with
individual bytes used by the kernel.

The shape can be misleading at first:

```text
gran_k = 32
one scale byte covers one 32-element K group
one torch.int / uint32_t word packs four scale bytes
therefore hidden / 128 int words represent hidden / 32 scale bytes
```

This is block-scaled quantization, but "block" here means a quantization group,
not a GEMM tile. In the MegaMoE FP8/FP4 path, the quantization block is:

```text
activation x:
  one token row x one 32-element K slice
  scale shape before packing: [num_tokens, ceil(hidden / 32)]

weight W:
  one output-channel row x one 32-element K slice
  scale shape before packing: [num_output_channels, ceil(K / 32)]
```

The Python reference helper for FP8 activations does:

```python
x_view = x_padded.view(m, padded_n // gran_k, gran_k)
x_amax = x_view.abs().float().amax(dim=2).clamp(1e-4)
sf = ceil_to_ue8m0(x_amax / 448.0)
x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
```

So each 32-element group is scaled independently:

```text
sf[m, g] = ceil_pow2(max(abs(x[m, 32g:32g+32])) / 448)
q[m, k]  = fp8_e4m3(x[m, k] / sf[m, floor(k / 32)])
x[m, k] ~= q[m, k] * sf[m, floor(k / 32)]
```

The Python helper for FP4 weights uses the same per-row / per-32-K pattern, but
the target FP4 E2M1 max finite value is 6 instead of 448:

```python
x_view = x_padded.view(m, -1, gran_k)
x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
sf = ceil_to_ue8m0(x_amax / 6.0)
x_scaled = x_view * (1.0 / sf.unsqueeze(2))
codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
codes2 = codes.view(m, padded_n // 2, 2)
packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
```

The FP4 E2M1 codebook used by the helper is:

```text
{0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}
```

Two FP4 codes are packed into one `int8` payload byte. Scale factors are
optionally packed with `pack_ue8m0_to_int`, which stores four UE8M0 scale bytes
inside one `int32` word. MegaMoE uses the packed scale-factor path for input
activations.

The kernel names activation scale factors as SFA and weight scale factors as
SFB when feeding UMMA:

```text
SFA: scale factors for A / activation tiles
SFB: scale factors for B / weight tiles
```

Weight SF layout is validated at the C++ API boundary:

```cpp
constexpr int kGranMN = 1, kGranK = 32;
check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden,
                kGranMN, kGranK, num_experts_per_rank, true, false,
                torch::kInt);
check_sf_layout(l2_weights_sf, hidden, intermediate_hidden,
                kGranMN, kGranK, num_experts_per_rank, true, false,
                torch::kInt);
```

So the FP8/FP4 path assumes one scale group per 32 values along K, and both L1
and L2 weight scale tensors have already been converted into the packed
integer layout expected by the CUDA kernel.

## Quantized Matmul Semantics

For one expert linear layer, the logical GEMM is:

```text
C[M, N] = A[M, K] @ B[N, K]^T
```

In the FP8/FP4 block-scaled path:

```text
A[m, k] ~= A_q[m, k] * A_sf[m, floor(k / 32)]
B[n, k] ~= B_q[n, k] * B_sf[n, floor(k / 32)]
```

So one output element is approximately:

```text
C[m, n] =
  sum over K-groups g:
    A_sf[m, g] * B_sf[n, g] *
      sum over k in group g:
        A_q[m, k] * B_q[n, k]
```

The important point is that the scale changes along the reduction dimension.
For every 32-wide K group, the hardware has one activation scale and one weight
scale for a given `(m, n)` output pair. The block-scaled UMMA instruction path
uses those scale factors as part of the matrix-multiply operation, rather than
running an explicit dequantize kernel that materializes BF16/FP32 A and B.

Concrete small example with `K = 128`:

```text
A_q[m, 0:128] has four scale bytes:
  A_sf[m, 0], A_sf[m, 1], A_sf[m, 2], A_sf[m, 3]

B_q[n, 0:128] has four scale bytes:
  B_sf[n, 0], B_sf[n, 1], B_sf[n, 2], B_sf[n, 3]

C[m, n] =
  A_sf[m,0] * B_sf[n,0] * dot(A_q[m,  0: 32], B_q[n,  0: 32])
  + A_sf[m,1] * B_sf[n,1] * dot(A_q[m, 32: 64], B_q[n, 32: 64])
  + A_sf[m,2] * B_sf[n,2] * dot(A_q[m, 64: 96], B_q[n, 64: 96])
  + A_sf[m,3] * B_sf[n,3] * dot(A_q[m, 96:128], B_q[n, 96:128])
```

Because four scale bytes pack into one `uint32_t`, the same `K = 128` example
also explains the public MegaMoE view shape:

```text
hidden / 128 uint32 words
  == hidden / 32 logical scale bytes
```

## Quantization Block Versus GEMM Tile

Do not conflate these two block sizes:

| Concept | Meaning | MegaMoE code anchor |
|---|---|---|
| quantization block | one row by 32 K elements sharing one scale | `gran_k = 32`, `kGranK = 32` |
| GEMM tile | a scheduled matrix multiply tile, e.g. `BLOCK_M x BLOCK_N x BLOCK_K` | kernel config / TMA / UMMA loops |
| ring block | reusable activation-buffer slot holding `BLOCK_M` token rows | dispatch / scheduling notes |

The quantization group is nested inside the GEMM K tile. For a GEMM tile:

```text
A payload tile:
  BLOCK_M rows x BLOCK_K columns

B payload tile:
  BLOCK_N rows x BLOCK_K columns

logical A scale bytes:
  BLOCK_M rows x (BLOCK_K / 32) K-groups

logical B scale bytes:
  BLOCK_N rows x (BLOCK_K / 32) K-groups

packed A/B scale words:
  rows x (BLOCK_K / 128) uint32 words
```

This is why the SM100 launch path uses:

```cpp
constexpr int kGranK = 32;
const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
```

and the kernel shared-memory scale buffers are sized like:

```cpp
smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)]
smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)]
```

The M/N tile dimensions decide how many token rows or output channels a CTA
cluster handles. The quantization block decides how many K elements share one
scale. They interact only through the scale-factor tile that must be loaded
alongside each payload tile.

For activation scales:

```text
SFA tile rows correspond to token rows in the A tile.
SF_BLOCK_M is BLOCK_M padded/aligned to 128 rows for the UTCCP copy path.
```

For weight scales:

```text
SFB tile rows correspond to output-channel rows in the B tile.
SF_BLOCK_N == BLOCK_N in the SM100 kernel.
```

So the scale-factor tile is smaller than the payload tile by a factor of 32
along K, and another factor of 4 in storage because four UE8M0 bytes are packed
into one `uint32_t`.

## Relation To DeepSeek-V3 FP8 Recipe

Do not merge this public SM100 MegaMoE FP8/FP4 kernel ABI with the DeepSeek-V3
FP8 training recipe.

DeepSeek-V3's paper describes FP8 training with:

```text
activation scaling:
  1 x 128 tile
  per token per 128 channels

weight scaling:
  128 x 128 block
  per 128 input channels per 128 output channels
```

The public DeepGEMM SM100 MegaMoE FP8/FP4 path described in this note uses:

```text
input activation payload:
  FP8 E4M3
  one scale per token per 32 K elements

intermediate activation payload:
  FP8 E4M3
  Linear1 epilogue re-quantizes one scale per token per 32 K elements

weight payload:
  FP4 E2M1
  separate path; do not compare directly to V3's FP8 weight recipe without
  calling out the dtype and hardware change
```

So, if comparing activation scale granularity only:

```text
V3 paper FP8 training activation:
  1 scale / 128 activation values

public SM100 MegaMoE FP8 activation path:
  1 scale / 32 activation values
```

That is finer-grained. But the precise claim should be:

```text
The public DeepGEMM SM100 FP8/FP4 MegaMoE path uses a finer activation
scale granularity than the DeepSeek-V3 FP8 training recipe.
```

It is not the same as saying:

```text
The V4 paper's entire training quantization recipe is simply "V3 but finer".
```

After checking the V4 paper, the correct boundary is:

```text
V4 FP4 QAT for MoE expert weights:
  post-training stage
  FP32 master weights -> FP4 quantized weights -> dequantized back to FP8
  computation reuses the existing FP8 training framework
  backward gradients flow through the FP8 weights and update FP32 masters

V4 inference and RL rollout:
  no backward pass
  native FP4 quantized weights are used directly
```

The paper is explicit that the FP4-to-FP8 dequantization can be lossless when
the ratio between max/min scale factors of FP4 sub-blocks inside each FP8 block
stays below a threshold, and it gives the relevant granularities:

```text
FP4 sub-block:
  1 x 32 tiles

FP8 quantization block:
  128 x 128 tiles
```

So the public MegaMoE kernel's `1 x 32` FP4 / scale granularity is not an
accidental implementation detail; it matches the V4 QAT description for FP4
expert weights. But the V4 paper also says that training computation still
goes through FP8 weights after dequantization, while native FP4 weights are
used directly for inference / RL rollout.

Also keep the kernel boundary straight:

```text
initial x quantization:
  caller-side setup before fp8_fp4_mega_moe

Linear1 output quantization:
  inside the MegaMoE kernel epilogue, producing l2_acts + l2_acts_sf
```

This difference lines up with the V3 paper's hardware discussion: V3 notes that
current Hopper-era Tensor Cores lack native support for their fine-grained
scaling and recommends future chips support group-scaled MMA. The public SM100
MegaMoE path is a concrete example of that direction: scale factors are loaded
through TMA / UTCCP and consumed by block-scaled UMMA.

The hardware interpretation is therefore:

```text
V3 on Hopper / H800:
  fine-grained scale is mostly a software / kernel-level workaround
  scale application and higher-precision accumulation require extra movement
  between Tensor Cores and CUDA cores
  1 x 128 activation and 128 x 128 weight blocks are a training/performance
  compromise

V4 public SM100 MegaMoE path:
  block-scaled UMMA can consume scale factors in the tensor-core path
  per-32 K-group scale is aligned with Blackwell-style microscaling support
  the extra metadata/layout cost is lower because scale factors are loaded
  through TMA / UTCCP and consumed by UMMA
```

The V4 MegaMoE section also frames the open-sourced MegaMoE performance
numbers as inference-oriented: it reports speedups for general inference
workloads, RL rollouts, and high-speed agent serving. That is why this note
keeps the public DeepGEMM forward kernel separate from claims about a full
training backward implementation.

## Why FP8 E4M3, FP4 E2M1, And UE8M0

The public SM100 FP8/FP4 MegaMoE kernel hard-codes these payload types:

```cpp
// activations are FP8 (e4m3), weights are FP4 (e2m1)
using a_dtype_t = cutlass::float_e4m3_t;
using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
```

The UMMA descriptor uses `cutlass::float_ue8m0_t` as the scale-factor type:

```cpp
cute::UMMA::make_instr_desc_block_scaled<
    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t, ...>()
```

The practical reason for this combination is:

- FP8 E4M3 gives activations one byte per value while keeping more mantissa
  precision than E5M2. With per-32-K scaling, the smaller dynamic range is
  acceptable; the helper uses 448 as the E4M3 finite-range target.
- FP4 E2M1 gives weights half a byte per value. The helper uses the E2M1
  finite values up to 6, so weight scale is based on `amax / 6`.
- UE8M0 stores scale factors as unsigned exponent-only powers of two. The
  helper rounds `amax / max_value` up to a power of two so the scaled payload
  should not overflow the target type.
- Power-of-two scales are cheap for hardware to apply and compact to store:
  one scale byte covers 32 payload elements, and four scale bytes pack into
  one `int32`.
- The SM100 block-scaled UMMA path has direct support for these payload and
  scale-factor types, so the kernel can keep dequantization fused into tensor
  core execution instead of materializing dequantized operands.

This is a hardware-aligned quantization format, not just a storage compression
choice. The scale layout, UTCCP transpose, TMA scale descriptors, and UMMA
scale IDs are all arranged so the tensor core instruction can consume payloads
and scale factors together.

## Kernel Boundary

The public `fp8_fp4_mega_moe` kernel does not take BF16 hidden states and
quantize them internally. The input quantization shown in `tests/test_mega_moe.py`
is caller-side setup:

```text
outside MegaMoE kernel:
  BF16 hidden states
    -> per_token_cast_to_fp8(...)
    -> copy FP8 payload to buffer.x
    -> copy packed scale factors to buffer.x_sf

inside MegaMoE kernel:
  read buffer.x + buffer.x_sf
    -> pull them through symmetric memory into l1_acts + l1_acts_sf
    -> feed Linear1 UMMA
```

The same boundary applies to weights:

```text
outside MegaMoE kernel:
  BF16 expert weights
    -> per_token_cast_to_fp4(...)
    -> transform_sf_into_required_layout(...)
    -> transform_weights_for_mega_moe(...)

inside MegaMoE kernel:
  consume transformed FP4 weights + transformed weight SF
```

Production code does not have to use the exact Python helpers from the test,
but it must satisfy the same ABI: before launching the FP8/FP4 MegaMoE kernel,
the symmetric buffer already contains quantized `x` and `x_sf`, and the weight
arguments already carry transformed FP4 weights and scale factors.

## Where Scale Factors Come From

The test path constructs quantized inputs before calling MegaMoE. This is
outside the fused kernel:

```python
x = per_token_cast_to_fp8(
    x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
buffer.x[:num_tokens].copy_(x[0])
buffer.x_sf[:num_tokens].copy_(x[1])
```

So, for inputs:

```text
BF16 x
  -> FP8 E4M3 payload x[0]
  -> packed UE8M0 scale metadata x[1]
  -> copied into symmetric-buffer views buffer.x and buffer.x_sf
  -> only then call fp8_fp4_mega_moe(...)
```

Weights are also quantized before MegaMoE:

```python
w[i], w_sf[i] = per_token_cast_to_fp4(
    bf16_weights[i], use_ue8m0=True, gran_k=32)
w_sf = deep_gemm.transform_sf_into_required_layout(
    w_sf, n, k, (1, 32), num_groups)
```

Then `transform_weights_for_mega_moe` applies MegaMoE-specific layout changes:

```python
# FP8/FP4 path:
l1_w = _interleave_weights(l1_weights[0])
l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
```

For BF16 MegaMoE, only the L1 gate/up weight interleave remains. There is no
`_sf` path:

```python
# BF16 path:
l1_transformed = _interleave_weights(l1_weights)
l2_transformed = l2_weights
```

## Why L1 Weight SF Is Interleaved

SwiGLU's first linear logically produces two streams:

```text
W1(x) = [gate(x), up(x)]
SwiGLU(x) = silu(gate(x)) * up(x)
```

DeepGEMM interleaves L1 gate/up weights in groups of 8:

```python
# [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...]
```

This interleave is along the L1 output-channel dimension, not along the hidden
/ K dimension. The L1 weight tensor shape is roughly:

```text
[num_local_experts, intermediate_hidden * 2, hidden]
```

For each expert, the second dimension starts as:

```text
original output-row layout:
  [gate_0, gate_1, ..., gate_{I-1},
   up_0,   up_1,   ..., up_{I-1}]
```

After `_interleave_weights(..., gran=8)`, it becomes:

```text
interleaved output-row layout:
  [gate_0..7,   up_0..7,
   gate_8..15,  up_8..15,
   gate_16..23, up_16..23,
   ...]
```

So "group of 8" means: take 8 gate output rows, then the corresponding 8 up
output rows, then continue with the next 8. It does not merge rows together.

Small example with `intermediate_hidden = 16`:

```text
original:
  gate0 gate1 ... gate15 up0 up1 ... up15

interleaved:
  gate0 ... gate7 up0 ... up7 gate8 ... gate15 up8 ... up15
```

The point is locality in the Linear1 epilogue. SwiGLU needs paired `gate_i` and
`up_i` values:

```text
out_i = silu(gate_i) * up_i
```

If the physical layout stayed `[all gate | all up]`, the epilogue would pair
values from two far-apart row ranges. Interleaving small gate/up groups makes
the values the epilogue needs appear close together in the TMEM/load/store
layout.

The L1 weight scale factors are interleaved the same way because the UMMA path
and epilogue read values in that physical order. If the weight payload and
weight scale factors were transformed differently, the block-scaled GEMM would
apply the wrong scale to the matching FP4 values.

L2 weights do not have a gate/up pair, so only the scale-factor transpose is
needed for L2.

## UTCCP Scale-Factor Layout

The SM100 kernel defines the scale-factor row transform:

```cpp
const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
    const uint32_t idx = token_idx_in_expert % BLOCK_M;
    return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
           (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
};
```

The Python helper performs the same style of 4-by-32 transpose for weight scale
factors before the kernel sees them:

```python
result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
            .transpose(2, 3)
            .reshape(num_groups, mn, packed_sf_k))
```

Interpretation:

```text
normal row order inside a 128-row group:
  0, 1, 2, ..., 31, 32, ..., 63, 64, ..., 95, 96, ..., 127

UTCCP-facing order:
  rows are rearranged as a 4 x 32 / 32 x 4 pattern
```

This layout is for the Blackwell SM100 scale-factor copy path:

```cpp
using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
```

So the transpose is not a semantic MoE operation. It is a hardware-facing
layout rewrite so the scale factors can be copied efficiently from shared
memory into TMEM for UMMA's block-scaled instructions.

## Dispatch Pull Carries Payload And SF

Dispatch pull copies token payload and input SF as two paired streams.

The payload path uses TMA-style bulk movement through a temporary pull buffer:

```cpp
ptx::tma_load_1d(
    pull_buffer.get_base_ptr(),
    math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
    pull_mbarrier, kNumBytesPerPull);

ptx::tma_store_1d(
    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
    pull_buffer.get_base_ptr(), kNumBytesPerPull);
```

The input scale-factor path then directly copies packed `uint32_t` scale
metadata from the source rank's `input_sf_buffer` into the destination rank's
`l1_sf_buffer`:

```cpp
constexpr uint32_t kNumSFUint32 = kHidden / 128;
const auto remote_sf_ptr = sym_buffer.map(
    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
    current_rank_in_expert_idx);
const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();

const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
    transform_sf_token_idx(token_idx_in_block);

if (j < kNumSFUint32)
    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
```

That means `l1_acts` and `l1_acts_sf` become ready together at the ring-block
level. Linear1 does not recompute input scales; it consumes the pulled scales
that came with the already-quantized input token.

## UMMA Consumes SFA And SFB

For each Linear1 / Linear2 GEMM block, one warp role TMA-loads activations and
activation scale factors:

```cpp
tma::copy<BLOCK_K, LOAD_BLOCK_M, ...>(
    tensor_map_a_ptr, ..., shared_storage.smem_a[stage_idx], k_idx, ring_m_idx, 2);
tma::copy<SF_BLOCK_M, 1, 0>(
    tensor_map_sfa_ptr, ..., shared_storage.smem_sfa[stage_idx], sfa_ring_m_idx, sfa_k_idx, 2);
```

Another warp role TMA-loads weights and weight scale factors:

```cpp
tma::copy<BLOCK_K, LOAD_BLOCK_N, ...>(
    tensor_map_b_ptr, ..., shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
tma::copy<BLOCK_N, 1, 0>(
    tensor_map_sfb_ptr, ..., shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
```

The UMMA issue path creates a block-scaled instruction descriptor:

```cpp
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
    UMMA_M, UMMA_N,
    cute::UMMA::Major::K, cute::UMMA::Major::K>();
```

It copies SFA and SFB from shared memory into TMEM:

```cpp
cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
```

Then each UMMA instruction is issued with scale-factor IDs:

```cpp
const auto runtime_instr_desc =
    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);

ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
    b_desc, a_desc, accum_stage_idx * UMMA_N,
    ..., runtime_instr_desc,
    kTmemStartColOfSFB, kTmemStartColOfSFA);
```

So the scale-factor path is part of the GEMM instruction stream, not a separate
multiply kernel around GEMM.

## Linear1 Epilogue Re-Quantizes For Linear2

Linear1 accumulates into TMEM, then the epilogue reads accumulator values,
applies SwiGLU and top-k weights, and produces Linear2 input.

The first reduction computes amax over the epilogue's activation values:

```cpp
thread_local_amax.x = max(thread_local_amax.x, abs(activation_values[i][k].x));
thread_local_amax.y = max(thread_local_amax.y, abs(activation_values[i][k].y));

amax_values[i].x = math::warp_reduce<4, true>(
    thread_local_amax.x, math::ReduceMax<float>());
```

The code also reduces across a warp pair through shared memory:

```cpp
shared_storage.amax_reduction[epilogue_warp_idx][...] = amax_values[i];
...
const float2 wp_amax =
    shared_storage.amax_reduction[epilogue_warp_idx ^ 1][...];
amax_values[i].x = max(amax_values[i].x, wp_amax.x);
```

Then it computes an E4M3-compatible scale and inverse scale:

```cpp
float2 sf, sf_inv;
math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);
```

`math.cuh` implements the UE8M0 scale as a power of two:

```cpp
const float2 finfo_factor = {1.0 / 448.0, 1.0 / 448.0};
const auto scaled = amax * finfo_factor;
const auto exp = ceil_log2(scaled);
sf = 2^exp;
sf_inv = 2^-exp;
```

The epilogue multiplies by `sf_inv`, casts to FP8 E4M3, stores FP8 values into
shared memory, and uses TMA store to publish `l2_acts`:

```cpp
const float2 upper = activation_values[i][0] * sf_inv;
const float2 lower = activation_values[i][1] * sf_inv;
const auto fp8x4_values = __nv_fp8x4_e4m3(...);
ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
...
cute::SM90_TMA_STORE_2D::copy(
    &tensor_map_l1_output,
    shared_storage.smem_d.l1[...],
    out_n_idx,
    ring_m_idx + ...);
```

It writes the matching L2 activation scale factors into `l2_sf_buffer` as
packed UE8M0 bytes:

```cpp
const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
sf_base_ptr[sf_addr] =
    (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
sf_base_ptr[sf_addr + 4 * sizeof(uint32_t)] =
    (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
```

This is the only in-kernel activation re-quantization point in the current
forward path:

```text
Linear1 output after SwiGLU/top-k weight
  -> compute amax
  -> choose power-of-two UE8M0 scale
  -> FP8 E4M3 payload in l2_acts
  -> packed scale metadata in l2_acts_sf
```

## Linear2 Output

Linear2 uses `l2_acts + l2_acts_sf` as its activation input and FP4 weights
plus weight scale factors as its weight input. Its epilogue writes BF16 output
to the remote combine buffer:

```cpp
// L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
```

There is no `l3_acts_sf` or second output re-quantization path in this public
forward kernel.

## Scale-Factor Path Summary

```text
Before MegaMoE kernel:
  BF16 x
    -> per_token_cast_to_fp8
    -> buffer.x + buffer.x_sf

  BF16 W1/W2
    -> per_token_cast_to_fp4
    -> transform_sf_into_required_layout
    -> transform_weights_for_mega_moe
    -> l1/l2 weights + weight SF

MegaMoE kernel boundary:
  kernel receives pre-quantized input payload/SF and transformed FP4 weights/SF

Dispatch inside MegaMoE kernel:
  remote input token payload
    -> local l1_acts ring slot

  remote input x_sf
    -> local l1_acts_sf ring slot with transform_sf_token_idx layout

Linear1 GEMM:
  l1_acts + l1_acts_sf
  l1_weights + l1_weights_sf
    -> UMMA block-scaled FP8 x FP4 accumulation in TMEM

Linear1 epilogue:
  TMEM accumulator
    -> SwiGLU/top-k-weighted FP32 values
    -> amax
    -> UE8M0 scale + inverse scale
    -> FP8 l2_acts + packed l2_acts_sf

Linear2 GEMM:
  l2_acts + l2_acts_sf
  l2_weights + l2_weights_sf
    -> UMMA block-scaled FP8 x FP4 accumulation in TMEM

Linear2 epilogue:
  TMEM accumulator
    -> BF16 remote combine buffer
```

## Open Questions

- The first-pass conclusion is about forward inference-style MegaMoE. A
  backward path would need separate scale-factor handling for gradient inputs,
  weight gradients, and possible reduction order issues.
- The numerical equivalence target is currently the public legacy baseline in
  `tests/test_mega_moe.py`, which asserts bitwise equality. That is a strong
  forward-path test, but not a proof for all production shapes or training
  configurations.
