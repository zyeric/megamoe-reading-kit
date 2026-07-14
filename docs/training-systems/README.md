# Training Systems

Notes on large-scale training systems and training-time infrastructure:
parallelism strategies, checkpointing, optimizer/runtime behavior, data
loading at system boundaries, fault tolerance, profiling, and stability.

Use this directory for papers, tech reports, repos, and implementation writeups
whose main relevance is training execution rather than model quality alone.

Current notes:

- `deepseek-v4-megamoe-index.html` - GitHub Pages entry page for the MegaMoE
  reading kit, linking the guide, talk, long-form notes, claims index, and
  glossary.
- `deepseek-v4-megamoe-talk.html` - concise one-hour HTML talk deck for
  MegaMoE, with visual mental models, timing, speaker notes, and links to
  official GPU architecture references.
- `deepseek-v4-megamoe-notes.html` - generated long-form HTML reading snapshot
  built from the local MegaMoE and hardware markdown notes.
- `deepseek-v4-megamoe-reading-guide.md` - human-first top-down reading guide:
  MoE semantics, distributed EP reference, fused runtime protocol, CUDA
  lowering, and suggested reading tracks.
- `deepseek-v4-megamoe-context-map.md` - agent-oriented routing map for which
  MegaMoE file owns which question, concept, layer, and update responsibility.
- `deepseek-v4-megamoe-glossary.md` - layer-aware terminology index for MoE,
  EP, runtime protocol, scheduling, numerics, and CUDA / hardware terms.
- `deepseek-v4-megamoe-claims-index.md` - evidence ledger separating
  code-backed claims, paper-backed claims, inference, and open questions.
- `deepseek-v4-megamoe-source-snapshot.md` - source provenance and revalidation
  checklist for paper paths, DeepGEMM ref, and code files.
- `deepseek-v4-moe-megakernel.md` - focused DeepSeek-V4 reading plan for the
  fused MoE megakernel, communication overlap, and deterministic MoE training
  contract.
- `deepseek-v4-megamoe-runtime-protocol.md` - cross-stage runtime protocol for
  public DeepGEMM MegaMoE: symmetric-buffer addressability, pool / ring slots,
  counters, capacity, source-token metadata, and resource lanes.
- `deepseek-v4-megamoe-dispatch.md` - dispatch lowering notes for public
  DeepGEMM MegaMoE: route metadata, source-rank ordering, token pull into L1
  ring slots, `TokenSrcMetadata`, and `l1_full_count` publication.
- `deepseek-v4-megamoe-scheduling.md` - execution scheduling notes for public
  DeepGEMM MegaMoE: waves, pool blocks, ring blocks, persistent workers,
  pipeline counters, and small-batch / imbalance behavior.
- `deepseek-v4-megamoe-quantization.md` - FP8/FP4 quantization and
  scale-factor path notes for public DeepGEMM MegaMoE: input / weight scales,
  UTCCP layout, UMMA scale consumption, and Linear1 epilogue re-quantization.
- `deepseek-v4-megamoe-activation.md` - Linear1 activation epilogue notes for
  public DeepGEMM MegaMoE: TMEM gate/up reads, BF16-rounded SwiGLU, top-k
  weighting, FP8 requantization, `l2_acts_sf`, and `l2_full_count`.
- `deepseek-v4-megamoe-gemm.md` - shared Linear1/Linear2 GEMM notes for public
  DeepGEMM MegaMoE: TMA A/B/SFA/SFB loads, shared-memory K-tile stages,
  block-scaled UMMA, TMEM accumulation, and forward determinism boundaries.
- `deepseek-v4-megamoe-combine.md` - Linear2 epilogue and final combine notes
  for public DeepGEMM MegaMoE: source metadata, remote BF16 route write-back,
  pre-combine barrier, top-k-slot summation, and determinism boundaries.
