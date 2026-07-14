# DeepSeek-V4 MegaMoE Source Snapshot

Status: source provenance and revalidation checklist

Context metadata:

- Topic: external source provenance for MegaMoE notes.
- Layer tags: `evidence`.
- Owns: paper paths, repository refs, file list, and revalidation checklist.
- Does not own: mechanism explanation or confidence of derived claims.
- Agent entry: read before comparing notes against newer upstream code.

Recorded: 2026-07-14, Asia/Shanghai

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

## Scope

This file records the external sources that the MegaMoE notes rely on. It is
not a proof that every line of every note was read at one exact upstream commit.
The earlier reading happened iteratively against public DeepGEMM `main` files.

Use this file to re-pin and revalidate the notes when upstream code changes.

## Paper Sources

| Source | Local path / URL | Use |
|---|---|---|
| DeepSeek-V4 Technical Report | [`../sources/2606.19348v1-deepseek-v4.pdf`](../sources/2606.19348v1-deepseek-v4.pdf) | V4 architecture motivation, section 3.1 MegaMoE framing, section 5.3 numerics |
| DeepSeek-V3 Technical Report | <https://arxiv.org/html/2412.19437> | FP8 recipe comparison and historical baseline |

## Repository Sources

| Repo | Recorded ref | Use |
|---|---|---|
| DeepGEMM | `deepseek-ai/DeepGEMM` `main` = `1f6f3f378920ccb5cc036ef43eb3f5972e921713` on 2026-07-14 | Public MegaMoE forward code path and tests |
| DeepEP | <https://github.com/deepseek-ai/DeepEP> | Baseline / follow-up for standalone EP communication |
| CUTLASS / CuTe | <https://github.com/NVIDIA/cutlass> | Tensor-core / CuTe vocabulary, if needed |
| Triton | <https://github.com/triton-lang/triton> | Background for persistent-kernel and kernel-authoring comparisons |

Command used to record the DeepGEMM ref:

```bash
gitp ls-remote https://github.com/deepseek-ai/DeepGEMM.git refs/heads/main
```

Output:

```text
1f6f3f378920ccb5cc036ef43eb3f5972e921713 refs/heads/main
```

## DeepGEMM Files Mentioned By Notes

| File | Why it matters |
|---|---|
| `tests/test_mega_moe.py` | Public executable contract, default shapes, benchmark accounting |
| `deep_gemm/mega/__init__.py` | Python `SymmBuffer` setup, input views, wrappers |
| `csrc/apis/mega.hpp` | Symmetric-buffer size calculation and tensor view shapes |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | JIT-selected launch path |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | Block shapes, warp/thread layout, wave size, pull chunk size, GEMM stages |
| `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` | Workspace layout, counters, metadata arrays, `TokenSrcMetadata` |
| `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh` | Symmetric-buffer mapping details |
| `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` | Wave-local Linear1 / Linear2 scheduler state machine |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | Main public SM100 FP8/FP4 fused kernel implementation |
| `deep_gemm/include/deep_gemm/common/math.cuh` | Scale-factor and FP8 helper math |
| `deep_gemm/utils/math.py` | Python reference helpers for FP8 / FP4 / scale packing |
| `third-party/tilelang_ops/swiglu.py` | Standalone SwiGLU baseline / reference path |

## Revalidation Checklist

When revisiting these notes after upstream changes:

1. Record the new DeepGEMM SHA.
2. Check whether `tests/test_mega_moe.py` still exercises the same public
   forward-only contract.
3. Check whether `fp8_fp4_mega_moe` still receives pre-quantized `x` / `x_sf`
   and precomputed `topk_idx` / `topk_weights`.
4. Check whether `SymmBuffer` setup and `sym_buffer.map` semantics changed.
5. Check whether dispatch still records `TokenSrcMetadata` and publishes
   `l1_full_count`.
6. Check whether activation still applies top-k weight before Linear2.
7. Check whether Linear2 still writes BF16 route outputs into remote combine
   slots before final local reduction.
8. Check whether any public backward / training kernels were added.
9. Check whether SM90, SM100, or future Rubin paths use materially different
   quantization / scale-factor / TMEM assumptions.
10. Update [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md)
    for every changed conclusion.

## Known Provenance Caveat

The existing notes often say "public DeepGEMM `main` files" because they were
written during an exploratory reading pass. The recorded SHA above is now the
anchor for future revalidation. It should not be overread as proof that all
earlier code snippets were captured from exactly that SHA.

For public GitHub Pages publication, keep this caveat visible so readers do not
confuse a living-code reading note with an archival source-code paper.
