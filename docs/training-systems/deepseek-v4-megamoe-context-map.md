# DeepSeek-V4 MegaMoE Context Map

Status: navigation layer for humans and agents

Context metadata:

- Topic: agent routing and note ownership for MegaMoE reading context.
- Layer tags: `evidence`, `runtime-protocol`, `scheduling`, `cuda-model`.
- Owns: problem-to-note routing, ownership matrix, layer tags, and update
  protocol.
- Does not own: technical mechanism details or proof of individual claims.
- Agent entry: read first for any future MegaMoE task.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-reading-guide.md`](deepseek-v4-megamoe-reading-guide.md)
  - top-down human reading path.
- [`deepseek-v4-megamoe-glossary.md`](deepseek-v4-megamoe-glossary.md) -
  layer-aware term index.
- [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md)
  - code-backed claims, inference, and open questions.
- [`deepseek-v4-megamoe-source-snapshot.md`](deepseek-v4-megamoe-source-snapshot.md)
  - paper / repo / source-file snapshot.

## Purpose

This file is the first file an agent should read before using the detailed
MegaMoE notes. It routes a question to the right source note and records which
file owns which layer of the explanation.

The main invariant:

```text
Do not append every new fact to dispatch.md.
First classify the fact by layer, then write it in the owning note.
```

## Agent Quickstart

For future tasks about MegaMoE:

1. Read this file.
2. Read [`deepseek-v4-megamoe-source-snapshot.md`](deepseek-v4-megamoe-source-snapshot.md)
   to understand source provenance and what has not been pinned.
3. Read [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md)
   to separate code-backed facts from inference.
4. Jump to the specific owner note from the routing table below.
5. Use [`deepseek-v4-megamoe-glossary.md`](deepseek-v4-megamoe-glossary.md)
   when a term could belong to more than one layer.

## Problem Routing Table

| If the question is about... | Read first | Then read | Key terms |
|---|---|---|---|
| Why MegaMoE exists at all | `deepseek-v4-moe-megakernel.md` | `deepseek-v4-megamoe-reading-guide.md` | MoE, EP, execution rewrite |
| How to read the notes after a break | `deepseek-v4-megamoe-reading-guide.md` | `deepseek-v4-megamoe-glossary.md` | reading path, layers |
| Which file owns a concept | this file | `deepseek-v4-megamoe-glossary.md` | owner note, layer |
| Whether a claim is solid | `deepseek-v4-megamoe-claims-index.md` | owner note, source snapshot | evidence level |
| Which code / paper version was used | `deepseek-v4-megamoe-source-snapshot.md` | owner note code anchors | snapshot, revalidation |
| Dispatch / token movement | `deepseek-v4-megamoe-runtime-protocol.md` | `deepseek-v4-megamoe-dispatch.md`, `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-symmetric-memory.md` | symmetric buffer, pool, ring, pull |
| Wave scheduling | `deepseek-v4-megamoe-scheduling.md` | `deepseek-v4-megamoe-runtime-protocol.md`, `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md` | wave, pool block, ring block |
| FP8 / FP4 quantization | `deepseek-v4-megamoe-quantization.md` | `deepseek-v4-megamoe-gemm.md`, `deepseek-v4-megamoe-activation.md` | E4M3, E2M1, UE8M0, SFA/SFB |
| Linear1 / Linear2 GEMM | `deepseek-v4-megamoe-gemm.md` | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md`, `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md` | TMA, UMMA, TMEM, UTCCP |
| SwiGLU activation epilogue | `deepseek-v4-megamoe-activation.md` | `deepseek-v4-megamoe-quantization.md`, `deepseek-v4-megamoe-gemm.md` | gate/up, top-k weight, amax |
| Combine / write-back | `deepseek-v4-megamoe-combine.md` | `deepseek-v4-megamoe-dispatch.md`, `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-symmetric-memory.md` | TokenSrcMetadata, combine buffer, barrier |
| CUDA execution vocabulary | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-execution-model.md` | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md` | CTA, warp, warpgroup, SM |
| GPU memory vocabulary | `https://zyeric.github.io/gpu-hardware-notes/notes/gpu-memory-hierarchy.md` | `https://zyeric.github.io/gpu-hardware-notes/notes/cuda-symmetric-memory.md` | registers, shared, L2, HBM, peer memory |
| Public writeup / GitHub Pages | `deepseek-v4-megamoe-index.html` | `deepseek-v4-megamoe-talk.html`, `deepseek-v4-megamoe-notes.html` | public entry, talk, source notes |

## Ownership Matrix

| File | Owns | Does not own |
|---|---|---|
| `deepseek-v4-megamoe-reading-guide.md` | top-down reading path, shallow-to-deep story, diagrams in text form | line-by-line source facts |
| `deepseek-v4-megamoe-context-map.md` | agent routing, document ownership, update protocol | deep technical explanation |
| `deepseek-v4-megamoe-glossary.md` | term definitions and layer mapping | proof of claims |
| `deepseek-v4-megamoe-claims-index.md` | claim status, evidence level, open questions | complete code walkthrough |
| `deepseek-v4-megamoe-source-snapshot.md` | external source provenance and revalidation checklist | mechanism explanation |
| `deepseek-v4-moe-megakernel.md` | model-to-kernel lowering map and research framing | detailed stage mechanics |
| `deepseek-v4-megamoe-runtime-protocol.md` | symmetric buffer, pool, ring, counters, metadata, barriers | dispatch source-rank order or GEMM internals |
| `deepseek-v4-megamoe-scheduling.md` | waves, persistent workers, bubbles, scheduling units | symmetric-memory transport or quantization |
| `deepseek-v4-megamoe-dispatch.md` | route metadata push, source-rank order, L1 ring pull, `TokenSrcMetadata` production | generic ring-buffer theory or wave policy |
| `deepseek-v4-megamoe-quantization.md` | payload / scale-factor ABI and FP8/FP4 semantics | activation algorithm beyond dtype path |
| `deepseek-v4-megamoe-activation.md` | Linear1 epilogue, SwiGLU, top-k weight, L2 ring publish | GEMM body or final combine |
| `deepseek-v4-megamoe-gemm.md` | Linear1/Linear2 shared TMA/UMMA/TMEM machinery | dispatch / combine ownership |
| `deepseek-v4-megamoe-combine.md` | Linear2 remote write-back and final top-k-slot reduction | Linear2 GEMM internals |
| `https://zyeric.github.io/gpu-hardware-notes/` | reusable CUDA / GPU background | DeepSeek-specific model semantics |

## Layer Tags

Use these tags when adding sections or claim rows:

| Tag | Meaning |
|---|---|
| `algorithm` | MoE math semantics independent of devices |
| `distributed-ep` | route ownership across ranks |
| `runtime-protocol` | symmetric buffers, counters, barriers, pool/ring objects |
| `scheduling` | wave / worker / traversal policy |
| `stage-dispatch` | route metadata and token pull into L1 |
| `stage-gemm` | Linear1/Linear2 tiled GEMM body |
| `stage-activation` | Linear1 epilogue / SwiGLU / requantization |
| `stage-combine` | Linear2 write-back and final reduction |
| `numerics` | dtype, quantization, scale-factor path |
| `cuda-model` | CTA / warp / SM / TMA / UMMA / TMEM vocabulary |
| `hardware` | physical GPU / memory / interconnect facts |
| `evidence` | claim source, confidence, open question |

## Update Protocol

When adding or revising MegaMoE notes:

1. Classify the new fact with one or more layer tags.
2. Add the detail to the owner note, not to the first note that mentioned the
   term.
3. If the fact changes a high-level conclusion, update
   [`deepseek-v4-megamoe-claims-index.md`](deepseek-v4-megamoe-claims-index.md).
4. If the fact depends on a new code version, update
   [`deepseek-v4-megamoe-source-snapshot.md`](deepseek-v4-megamoe-source-snapshot.md).
5. If the term is likely to be confused later, add or update it in
   [`deepseek-v4-megamoe-glossary.md`](deepseek-v4-megamoe-glossary.md).
6. Regenerate [`deepseek-v4-megamoe-notes.html`](deepseek-v4-megamoe-notes.html)
   with `python3 repo_reading/training-systems/render_megamoe_notes.py`.

## Public Artifact Map

For GitHub Pages, keep three reading surfaces:

| Surface | File | Intended reader |
|---|---|---|
| Landing page | `deepseek-v4-megamoe-index.html` | someone deciding what to read |
| One-hour talk | `deepseek-v4-megamoe-talk.html` | human audience, presentation mode |
| Long source snapshot | `deepseek-v4-megamoe-notes.html` | humans and agents needing details |

The markdown files remain the source of truth. HTML files are reading artifacts.
