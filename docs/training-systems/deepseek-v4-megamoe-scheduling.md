# DeepSeek-V4 / DeepGEMM MegaMoE Scheduling Notes

Status: first-pass execution scheduling notes

Context metadata:

- Topic: wave-level execution scheduling for public MegaMoE forward.
- Layer tags: `scheduling`, `runtime-protocol`, `cuda-model`.
- Owns: waves, persistent workers, pool-block traversal, ready-work polling,
  bubbles, and imbalance intuition.
- Does not own: symmetric-memory transport, route metadata format, or
  block-scaled GEMM numerics.
- Agent entry: read after runtime protocol when the task mentions wave,
  persistent kernel, block readiness, or imbalance.

Parent note: [`deepseek-v4-moe-megakernel.md`](deepseek-v4-moe-megakernel.md)

Related notes:

- [`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md)
  - pool / ring-buffer definitions, counters, capacity, and cross-stage
  protocol edges.
- [`deepseek-v4-megamoe-dispatch.md`](deepseek-v4-megamoe-dispatch.md) -
  metadata push, deterministic source-rank order, and token pull into L1 ring
  slots.
- [`gpu-hardware-notes/notes/cuda-kernel-patterns.md`](https://zyeric.github.io/gpu-hardware-notes/notes/cuda-kernel-patterns.md)
  - generic persistent-kernel, wave-scheduling, ring-buffer, and spin-wait
  patterns.
- [`gpu-hardware-notes/notes/cuda-symmetric-memory.md`](https://zyeric.github.io/gpu-hardware-notes/notes/cuda-symmetric-memory.md)
  - peer-addressability and symmetric-memory runtime model.

Scope:

- Explain the high-level MegaMoE execution scheduler: waves, pool blocks, ring
  blocks, persistent workers, and pipeline dependencies.
- Keep dispatch transport details in the dispatch note and CUDA hardware
  details in the hardware notes.
- Current concrete reading target is public DeepGEMM MegaMoE forward. Treat
  training backward and non-public V4 production paths as follow-up work.

## Evidence Status

This note is a conceptual scheduling map from the current DeepGEMM MegaMoE code
reading. It should be used to guide code reading, not as a final line-by-line
source citation.

Code paths to pin when tightening this note:

- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` - persistent
  MegaMoE kernel, dispatch / TMA / MMA / epilogue roles, counters, and work
  traversal.
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` - workspace layout,
  receive counts, pool offsets, and ring-buffer counters.
- `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` - JIT-selected kernel
  configuration and launch path.
- `tests/test_mega_moe.py` - public shape defaults and benchmark contract.

## One-Line Model

Wave scheduling is a higher-level execution grouping over local experts. It is
not the same thing as a ring-buffer block.

```text
wave:
  scheduling window over a subset of local experts

pool block:
  logical compute block, usually BLOCK_M routed token rows for one expert

ring block:
  physical reusable buffer slot holding BLOCK_M token rows
```

The wave decides which expert work is exposed to the persistent worker pool.
The ring block decides where a particular token block temporarily lives and how
producer-consumer synchronization is enforced.

## Terminology

| Term | Layer | Meaning |
|---|---|---|
| `wave` | execution scheduling | A group of local experts / expert blocks considered together by the fused kernel scheduler |
| `local expert` | distributed MoE layout | Expert owned by the current rank |
| `recv_tokens[expert]` | dispatch metadata | Number of routed token-topk entries received for a local expert |
| `BLOCK_M` | GEMM/kernel shape | Number of token rows in one GEMM M block |
| `pool block` | logical work | One `BLOCK_M` chunk in the concatenated expert-token pool |
| `ring block` | physical workspace | One reusable `BLOCK_M` slot in the L1/L2 activation ring buffer |
| `l1_full_count` | producer-consumer sync | Dispatch has filled enough rows for a Linear1 ring block generation |
| `l1_empty_count` | producer-consumer sync | Linear1 has consumed a ring block generation, so dispatch can reuse it |
| `l2_full_count` | producer-consumer sync | Linear1 epilogue has produced Linear2 input for a later stage |

The canonical pool / ring / counter definitions live in
[`deepseek-v4-megamoe-runtime-protocol.md`](deepseek-v4-megamoe-runtime-protocol.md).
This scheduling note only explains how those objects participate in wave-level
execution.

## Where Wave Shape Comes From

The scheduler needs routing-derived metadata, especially how many tokens each
local expert receives. The rough flow is:

```text
top-k routes already exist
  -> dispatch metadata push counts routes per expert-owner rank
  -> metadata barrier makes expert receive counts visible
  -> local rank computes pool offsets / block counts for its experts
  -> fused scheduler traverses work in wave-sized windows
```

For one expert:

```text
num_m_blocks[expert] = ceil(recv_tokens[expert] / BLOCK_M)
```

The number and shape of waves are therefore runtime-shape-aware: they depend on
the current routing distribution and the selected kernel configuration. They
should not be read as a fixed mathematical property of the model.

But this is not a CPU-style dynamic scheduler that continuously observes SM
idle time and reshuffles tasks at runtime. A better mental model is:

```text
routing metadata + kernel config define a flattened execution plan
resident CTAs / workers walk that plan with deterministic strides
counters decide whether a claimed block is ready to execute
```

## Persistent Worker Pool

MegaMoE-style scheduling is closer to a persistent worker pool than to "one CUDA
block per logical work item".

Simplified pseudocode:

```cpp
// Pseudocode, not exact DeepGEMM source.
for (wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
    Wave wave = make_wave(local_experts, recv_counts, wave_idx);

    for (flat_block = worker_id;
         flat_block < wave.num_blocks;
         flat_block += num_resident_workers) {
        Task task = decode_wave_block(wave, flat_block);

        wait_until_input_ready(task);
        compute_task(task);
        publish_output_ready(task);
    }
}
```

Only resident CTAs / warps occupy hardware resources. Logical work that has not
yet been claimed by a resident worker is just scheduler state; it does not have
its own CTA spinning.

So this is the wrong mental model:

```text
every expert block has a dedicated compute unit waiting for readiness
```

The better model is:

```text
a limited worker pool repeatedly claims blocks from the current wave's work list
```

Spin-wait happens only after a resident worker has claimed a block whose
producer dependency is not ready yet.

## Wave, Pool Block, And Ring Block

The layers compose like this:

```text
wave
  -> contains a flattened list of expert pool blocks
      -> each pool block maps to a reusable physical ring block
```

The pool is logical:

```text
expert 0 token blocks
expert 1 token blocks
expert 2 token blocks
...
```

Each expert segment is rounded to `BLOCK_M` rows for GEMM scheduling:

```text
pool_block_idx = expert_pool_block_offset + m_block_idx_in_expert
```

The ring buffer is physical and finite:

```text
ring_block_idx = pool_block_idx % kNumRingBlocks
generation     = pool_block_idx / kNumRingBlocks
```

This is why `wave != ring block`. One wave can cover many pool blocks, and many
pool blocks can reuse the same physical ring block across generations.

Example:

```text
wave 0: experts 0, 1

expert 0 has pool blocks 0, 1
expert 1 has pool block  2

kNumRingBlocks = 2

pool block 0 -> ring block 0, generation 0
pool block 1 -> ring block 1, generation 0
pool block 2 -> ring block 0, generation 1
```

The scheduler sees three logical pool blocks in one wave. The physical ring
buffer has only two blocks, so ring block 0 is reused after Linear1 releases
its previous generation.

## Are There Wave Barriers?

Do not model wave boundaries as a large explicit global barrier after every
wave. The fused pipeline is mainly connected by readiness counters:

```text
dispatch -> Linear1:
  l1_full_count[ring_block] reaches the expected generation value

Linear1 -> dispatch ring reuse:
  l1_empty_count[ring_block] reaches the expected generation value

Linear1 -> Linear2:
  l2_full_count[...] says the intermediate activation block is ready

Linear2 / epilogue -> combine:
  output metadata and counters make the route output visible
```

There is an important phase-level exception before token pull:

```text
dispatch metadata push
  -> cross-rank / peer-visible metadata barrier
  -> token pull can safely read source metadata and receive counts
```

That barrier is about metadata visibility across ranks. It is not the same
thing as a wave-to-wave barrier inside the steady-state compute pipeline.

## What Actually Spins

The scheduler may expose a block before its producer has filled the required
ring slot. In that case the claimed worker spins on a counter:

```cpp
// Pseudocode.
while (ld_acquire(l1_full_count[ring_block]) != expected_full_value) {
    // spin
}
```

Or dispatch may want to reuse a physical ring block before Linear1 has released
the previous generation:

```cpp
// Pseudocode.
while (ld_acquire(l1_empty_count[ring_block]) < expected_empty_value) {
    // spin
}
```

This does not mean every future block is spinning. It means the specific worker
that claimed this block is waiting for a producer-consumer dependency.

## Why Waves Help Small / Imbalanced Experts

If one local expert receives very few tokens, it may produce too few GEMM
blocks to keep all SMs busy:

```text
blocks_for_expert_e = ceil(tokens_e / BLOCK_M) * num_n_blocks
```

If `blocks_for_expert_e < num_resident_workers`, many workers would have
nothing useful to do if the scheduler processed that expert alone.

Wave scheduling exposes multiple experts together:

```text
wave = experts [e0, e1, e2, ...]
work list =
  e0 m0 n0
  e0 m0 n1
  ...
  e1 m0 n0
  ...
```

The important property is not that experts are perfectly balanced. It is that
the flattened wave work list can contain enough similarly shaped blocks to keep
the worker pool occupied.

Because GEMM block shapes are mostly fixed after padding, individual block time
is more uniform than raw expert token counts. The wave therefore amortizes
expert-level imbalance into a larger block pool.

## Why Waves Are Not Unlimited

Making a wave too wide is not always better. A wider wave can expose more
blocks, but it can also hurt:

- locality for expert weights and metadata;
- ring-buffer pressure and generation reuse;
- counter pressure in L2;
- scheduling clarity when one expert is already large enough to fill the SMs.

Useful rule of thumb:

```text
small / long-tail batch:
  widen the wave to expose enough blocks across experts

large expert batch:
  narrower waves may be enough and can preserve locality / workspace behavior
```

This is why wave shape is a utilization-locality tradeoff, not just a
parallelism knob.

## Pipeline Bubbles

In the ideal steady state, there should not be a large wave-to-wave idle gap.
Workers finish blocks, advance through the current wave's flattened list, and
then move on according to the next scheduling window.

Some idle time is still expected:

- pipeline fill and drain at the beginning/end;
- wave tail effects when few blocks remain;
- workers that claimed blocks before `l1_full_count` / `l2_full_count` is
  ready;
- dispatch blocked by `l1_empty_count` because ring slots are not released;
- peer-memory, HBM, TMA, epilogue, or tensor-core imbalance;
- pathological routing skew where one expert dominates the work list.

So the intended conclusion is:

```text
wave scheduling reduces utilization loss from small or imbalanced experts;
it does not eliminate all bubbles or make the pipeline fully dynamic.
```

This matches why paper-level discussions often emphasize RL rollout and
latency-sensitive serving: those workloads can have long-tail small batches
where per-expert work is too small unless the scheduler groups experts into
larger waves.

## Reading Checklist

When reading a MegaMoE scheduling code path, write down:

- what metadata determines `recv_tokens[expert]`;
- how `BLOCK_M`, `num_m_blocks`, and `num_n_blocks` are chosen;
- how local experts are grouped into waves;
- how a flat block id decodes into `(expert, m_block, n_block, phase)`;
- which counters each phase waits on before computing;
- which counters each phase publishes after producing output;
- whether wave boundaries imply a real barrier or only scheduler order;
- how ring-buffer generations prevent overwrite-before-consume;
- what happens for empty experts and partial final `BLOCK_M` blocks.
