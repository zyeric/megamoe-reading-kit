# DeepSeek-V4 MegaMoE Claims Index

Status: evidence ledger for MegaMoE notes

Context metadata:

- Topic: claim confidence and evidence separation.
- Layer tags: `evidence`, `algorithm`, `distributed-ep`, `runtime-protocol`,
  `numerics`, `hardware`.
- Owns: code-backed claims, paper-backed claims, hardware-doc-backed claims,
  inference, and open questions.
- Does not own: full source walkthroughs or public landing-page narrative.
- Agent entry: read before answering whether a MegaMoE statement is proven,
  inferred, or still open.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-source-snapshot.md`](deepseek-v4-megamoe-source-snapshot.md)
  - source provenance and revalidation checklist.
- [`deepseek-v4-megamoe-context-map.md`](deepseek-v4-megamoe-context-map.md)
  - where to route follow-up work.

## Evidence Levels

| Level | Meaning | How to use |
|---|---|---|
| Code-backed | Tied to public DeepGEMM files / tests already read in these notes | Can be used as a working fact for the recorded source path |
| Paper-backed | Tied to the local DeepSeek-V4 or V3 technical reports | Use for motivation / reported results; do not infer code details |
| Hardware-doc-backed | Tied to NVIDIA / CUDA documentation | Use for generic GPU concepts, not DeepGEMM policy |
| Inference | Reasonable conclusion from code shape, naming, comments, or execution model | State as inference and revalidate before relying on it |
| Open | Not proved by current notes | Do not treat as fact |

## Code-Backed Claims

| Claim | Evidence pointer | Confidence | Owner |
|---|---|---|---|
| The notes currently target public DeepGEMM SM100 FP8/FP4 MegaMoE forward path. | Stage notes list SM100 FP8/FP4 code anchors and tests. | High | source snapshot |
| Router/top-k computation is outside the public fused MegaMoE kernel path described here. | Dispatch note kernel-boundary section: `x`, `x_sf`, `topk_idx`, and `topk_weights` are available before dispatch pull. | High | dispatch |
| Input hidden states are quantized before the MegaMoE kernel ABI. | Quantization note tracks `per_token_cast_to_fp8(..., gran_k=32)` in the test / wrapper path. | High | quantization |
| Dispatch output is local Linear1 ring input plus source metadata, not final `y`. | Dispatch note one-line model and boundary-to-Linear1 section. | High | dispatch |
| `TokenSrcMetadata` records source ownership for later write-back. | Dispatch and combine notes connect metadata to Linear2 epilogue. | High | dispatch / combine |
| Pool and ring are distinct: pool is logical work, ring is reusable physical storage. | Runtime protocol note pool/ring model and counter example. | High | runtime protocol |
| Ring slot reuse is protected by generation counters. | Runtime protocol producer-consumer counter section. | High | runtime protocol |
| Wave scheduling is higher-level than ring-block reuse. | Scheduling note one-line model and terminology table. | High | scheduling |
| Linear1 and Linear2 share the same high-level TMA / UMMA / TMEM GEMM machinery. | GEMM note treats them together and maps tensor differences. | High | GEMM |
| Scale-factor metadata is consumed by block-scaled UMMA, with UTCCP involved for SF metadata movement. | GEMM and quantization notes on SFA/SFB and UTCCP. | Medium-high | GEMM / quantization |
| Linear1 epilogue applies gate/up SwiGLU, top-k weight, amax, scale, and FP8 requantization for Linear2 input. | Activation and quantization notes. | High | activation |
| Top-k weight is applied before Linear2 in the public FP8/FP4 path, so final combine sums preweighted route outputs. | Activation note and combine note. | High | activation / combine |
| Linear2 epilogue writes BF16 route outputs into remote token-owner combine slots. | Combine note remote write-back section. | High | combine |
| Final combine uses a barrier before local top-k-slot reduction. | Combine note pre-combine barrier section. | High | combine |

## Paper-Backed Claims

| Claim | Evidence pointer | Confidence | Owner |
|---|---|---|---|
| V4 report presents MegaMoE speedups for inference-like workloads, including RL rollout and high-speed agent serving. | Local DeepSeek-V4 PDF section 3.1, as summarized in quantization / overview notes. | High | megakernel overview |
| Public MegaMoE discussion should not be automatically treated as proof of a full public training backward implementation. | V4 report wording plus lack of pinned public backward path. | Medium-high | claims index |
| V3 FP8 recipe used coarser FP8 scale-factor framing than the public V4 MegaMoE FP8/FP4 path described here. | DeepSeek-V3 report section 3.3.2 / appendix B.2 and quantization note comparison. | Medium | quantization |

## Hardware-Doc-Backed Claims

| Claim | Evidence pointer | Confidence | Owner |
|---|---|---|---|
| `sm90` / `sm100` identify architecture generations, not a GPU's SM count. | CUDA / NVIDIA tuning guide background summarized in hardware notes. | High | hardware |
| CTA, warp, and SM live at different layers: programming abstraction, scheduling unit, and physical execution block. | GPU execution model note. | High | hardware |
| TMA and TMEM are different concepts: TMA moves tensor data, TMEM holds tensor-core accumulator state on Blackwell. | GPU memory hierarchy and kernel pattern notes. | High | hardware |
| L1/shared memory, L2 cache, and HBM have different scope and management models. | GPU memory hierarchy note. | High | hardware |

## Inferences To Revalidate

| Inference | Current reasoning | Risk | Revalidation |
|---|---|---|---|
| Public MegaMoE path is best read as NVLink-domain peer memory, not a cross-node RDMA all-to-all stack. | Symmetric memory, `nvlink_barrier`, peer mapping, and benchmark framing point this way. | Medium | Audit PyTorch symmetric-memory backend and DeepGEMM launch requirements at pinned SHA. |
| Pathological route imbalance mainly hurts tail waves / ring occupancy rather than changing correctness. | Scheduling model and capacity discussion imply this. | Medium | Test with synthetic imbalanced `topk_idx` once GPU access is available. |
| Backward could conceptually reuse some forward protocol objects but would need separate dX / dW reduction design. | MoE backward structure and missing public code. | Medium | Revisit if public backward kernels appear. |
| Some UTCCP / TMEM details are Blackwell-specific and should not be generalized to Hopper. | SM100 implementation and Blackwell tensor-memory terminology. | Medium | Compare SM90 public path or CUTLASS/CuTe docs if available. |

## Open Questions

| Question | Why it matters | Owner |
|---|---|---|
| Is there a public backward / training MegaMoE kernel path? | Determines whether current notes are inference-only or training-relevant. | source snapshot |
| What exact policy chooses `num_max_tokens_per_rank` in production serving or training? | Capacity and imbalance behavior depend on it. | runtime protocol |
| How does the implementation behave under extreme imbalance beyond examples in the public benchmark? | Determines robustness of wave scheduling claims. | scheduling |
| What exact source block implements the deterministic source-rank interleaving order? | Would tighten dispatch determinism evidence. | dispatch |
| Which PTX / SASS instructions implement the low-level UMMA / UTCCP / TMEM details? | Needed for architecture-portability claims. | GEMM / hardware |
| Does symmetric memory in this stack support only one NVLink domain, or can it be composed with other communication layers? | Determines relationship to DeepEP / cross-node EP. | symmetric memory |

## How To Add A New Claim

1. Add the claim to the strongest applicable table.
2. Link to the owner note and source snapshot if possible.
3. If confidence is below high, describe exactly what would prove it.
4. If the claim changes the public HTML story, regenerate
   [`deepseek-v4-megamoe-notes.html`](deepseek-v4-megamoe-notes.html).
