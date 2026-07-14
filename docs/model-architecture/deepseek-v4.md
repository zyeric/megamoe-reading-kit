# DeepSeek-V4 Technical Report

Status: active deep dive

Sources:

- Local PDF: [`../sources/2606.19348v1-deepseek-v4.pdf`](../sources/2606.19348v1-deepseek-v4.pdf)
- PDF: https://arxiv.org/pdf/2606.19348v1
- Searchable HTML: https://arxiv.org/html/2606.19348v1

Local PDF metadata:

- Size: 4.5 MB
- SHA256: `55b2d72f772ac00de2e470b3ee08443c648d971c7f57c52d6202895665e5978d`

Paper title: `DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence`

This note is the working entrypoint for reading the DeepSeek-V4 report. It
starts under `model-architecture/` because architecture is the main lens, but
the intended read is cross-system: model architecture, training system,
inference system, and RL/post-training system.

## Reading Goals

- Identify the model architecture changes that matter for system behavior:
  attention, context length, memory layout, MoE/routing if present, positional
  encoding, auxiliary modules, or any special long-context mechanism.
- Extract training-system implications: parallelism strategy, sequence packing
  or varlen requirements, activation/checkpoint/memory pressure, optimizer or
  precision assumptions, and expected communication patterns.
- Extract inference-engine implications: prefill/decode shape, KV cache
  footprint, million-token serving constraints, chunking/prefix reuse, batching,
  speculative paths, and evaluation/serving throughput assumptions.
- Extract RL/post-training implications: rollout requirements, verifier/reward
  model dependencies, sampling strategy, long-context RL/eval workload shape,
  and any training-loop changes.

## Initial Questions

- What is the core mechanism behind the million-token context claim?
- Does the architecture require new attention kernels, ring/context-parallel
  strategies, compression, retrieval-like state, or special KV cache handling?
- How much of the result depends on training recipe versus inference-time
  system design?
- What parts would be hard to reproduce without DeepSeek's data, cluster, or
  serving stack?
- Which claims are benchmark/evaluation claims, and what caveats should be
  tracked before treating them as reusable lessons?

## Reading Template

Use this structure while reading:

```text
Claim:
Mechanism:
Evidence:
System implication:
Benchmark/eval caveat:
Reusable lesson:
Open question:
```

## Cross-Area Notes To Split Later

- Training-system notes should move or link to
  `repo_reading/training-systems/` if they become substantial.
- Focused training-system deep dive:
  [`../training-systems/deepseek-v4-moe-megakernel.md`](../training-systems/deepseek-v4-moe-megakernel.md).
- Inference-engine notes should move or link to
  `repo_reading/inference-engines/`.
- RL/post-training notes should move or link to
  `repo_reading/rl-posttraining/`.
- Data/eval caveats should move or link to
  `repo_reading/data-eval-watchlist/`.

## Source Artifact Policy

The arXiv PDF is public and has been committed under `repo_reading/sources/`
for shared offline reading. Keep the arXiv links above as canonical upstream
references.
