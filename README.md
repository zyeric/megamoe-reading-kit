# DeepSeek-V4 MegaMoE Reading Kit

Public reading kit for understanding the DeepSeek-V4 / DeepGEMM MegaMoE
forward path from MoE semantics down to GPU-side execution protocol.

The current material focuses on the public DeepGEMM SM100 FP8/FP4 MegaMoE
forward path. It separates code-backed facts, paper-backed claims, inference,
and open questions so the notes can serve both human readers and future agents.

## Published Site

GitHub Pages should publish from:

```text
branch: main
folder: /docs
```

After Pages is enabled, the site should be available at:

```text
https://zyeric.github.io/megamoe-reading-kit/
```

## Reading Surfaces

- `docs/index.html` - public landing page.
- `docs/training-systems/deepseek-v4-megamoe-talk.html` - one-hour talk deck.
- `docs/training-systems/deepseek-v4-megamoe-notes.html` - long-form reading
  snapshot with guide, source notes, and diagrams.
- `docs/training-systems/deepseek-v4-megamoe-context-map.md` - agent routing
  map and document ownership.
- `docs/training-systems/deepseek-v4-megamoe-claims-index.md` - evidence
  ledger.
- `docs/training-systems/deepseek-v4-megamoe-glossary.md` - layer-aware
  terminology.

## Repository Layout

```text
docs/
  index.html
  training-systems/
  model-architecture/
  sources/
```

The reusable hardware background notes are maintained separately at:

```text
https://zyeric.github.io/gpu-hardware-notes/
```

MegaMoE pages link to that site instead of vendoring the full hardware notes.

## Provenance

This repo was split from the private working context in
`axis-training-dev-tools`. The public source snapshot is recorded in:

```text
docs/training-systems/deepseek-v4-megamoe-source-snapshot.md
```

## License

License is not selected yet. Decide on an explicit content/code license before
promoting broad reuse.
