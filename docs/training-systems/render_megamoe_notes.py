#!/usr/bin/env python3
"""Render the DeepSeek-V4 MegaMoE reading notes into one static HTML file.

The repo markdown files remain the source of truth. This script is intentionally
small and dependency-free so the reading snapshot can be regenerated on a
machine without pandoc or Python markdown packages.
"""

from __future__ import annotations

import datetime as dt
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE = Path(__file__).resolve().parent
REPO_READING = BASE.parent
OUTPUT = BASE / "deepseek-v4-megamoe-notes.html"
HARDWARE_SITE = "https://zyeric.github.io/gpu-hardware-notes/notes/"


@dataclass(frozen=True)
class SourceDoc:
    group: str
    path: Path
    label: str
    summary: str


@dataclass(frozen=True)
class ExternalDoc:
    label: str
    url: str
    summary: str


GUIDE_DOCS = [
    SourceDoc(
        "Guide",
        BASE / "deepseek-v4-megamoe-reading-guide.md",
        "Reading Guide",
        "Human-first path from MoE semantics to GPU-side execution protocol.",
    ),
    SourceDoc(
        "Guide",
        BASE / "deepseek-v4-megamoe-context-map.md",
        "Context Map",
        "Agent routing table, note ownership, layer tags, and update protocol.",
    ),
    SourceDoc(
        "Guide",
        BASE / "deepseek-v4-megamoe-glossary.md",
        "Glossary",
        "Layer-aware definitions for MegaMoE, CUDA, numerics, and runtime terms.",
    ),
    SourceDoc(
        "Guide",
        BASE / "deepseek-v4-megamoe-claims-index.md",
        "Claims Index",
        "Code-backed claims, inference, paper-backed claims, and open questions.",
    ),
    SourceDoc(
        "Guide",
        BASE / "deepseek-v4-megamoe-source-snapshot.md",
        "Source Snapshot",
        "Paper, repo, file, and revalidation provenance for the notes.",
    ),
]


CORE_DOCS = [
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-moe-megakernel.md",
        "Lowering Map",
        "Algorithm semantics, EP reference, fused execution rewrite, and coverage.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-runtime-protocol.md",
        "Runtime Protocol",
        "Symmetric buffer, pool/ring slots, counters, source metadata, and resource lanes.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-scheduling.md",
        "Scheduling",
        "Waves, pool blocks, ring blocks, persistent workers, and bubbles.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-dispatch.md",
        "Dispatch",
        "Route metadata, source-rank order, L1 ring pull, and readiness publication.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-quantization.md",
        "Quantization",
        "FP8 / FP4 payloads, UE8M0 scale factors, and kernel ABI boundaries.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-activation.md",
        "Activation",
        "Linear1 epilogue, BF16-rounded SwiGLU, top-k weight, and L2 ring output.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-gemm.md",
        "GEMM",
        "Linear1 / Linear2 shared tiled GEMM body, TMA, UTCCP, UMMA, and TMEM.",
    ),
    SourceDoc(
        "Core",
        BASE / "deepseek-v4-megamoe-combine.md",
        "Combine",
        "Linear2 remote write-back, pre-combine barrier, and local top-k reduction.",
    ),
]


HARDWARE_DOCS = [
    ExternalDoc(
        "Symmetric Memory",
        HARDWARE_SITE + "cuda-symmetric-memory.md",
        "Peer addressability, rendezvous, hot-path pointer mapping, and caveats.",
    ),
    ExternalDoc(
        "GPU Execution Model",
        HARDWARE_SITE + "gpu-execution-model.md",
        "CTA / warp / SM vocabulary, persistent kernels, PTX, and SASS.",
    ),
    ExternalDoc(
        "GPU Memory",
        HARDWARE_SITE + "gpu-memory-hierarchy.md",
        "Registers, shared memory, TMEM, L2, HBM, TMA, and locality.",
    ),
    ExternalDoc(
        "Kernel Patterns",
        HARDWARE_SITE + "cuda-kernel-patterns.md",
        "Tiling, ring buffers, counters, wave scheduling, and fused pipelines.",
    ),
]


class MarkdownRenderer:
    def __init__(self, collect_toc: bool = True) -> None:
        self.collect_toc = collect_toc
        self.used_ids: dict[str, int] = {}
        self.toc: list[tuple[int, str, str, str]] = []

    def slugify(self, text: str) -> str:
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text).lower()
        slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        if not slug:
            slug = "section"
        seen = self.used_ids.get(slug, 0)
        self.used_ids[slug] = seen + 1
        return slug if seen == 0 else f"{slug}-{seen + 1}"

    def inline(self, text: str) -> str:
        code_spans: list[str] = []

        def keep_code(match: re.Match[str]) -> str:
            code_spans.append(f"<code>{html.escape(match.group(1))}</code>")
            return f"\x00CODE{len(code_spans) - 1}\x00"

        text = re.sub(r"`([^`]+)`", keep_code, text)
        text = html.escape(text)

        def link(match: re.Match[str]) -> str:
            label = match.group(1)
            href = match.group(2)
            return f'<a href="{html.escape(href, quote=True)}">{label}</a>'

        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        for idx, rendered in enumerate(code_spans):
            text = text.replace(f"\x00CODE{idx}\x00", rendered)
        return text

    def heading(self, level: int, text: str, doc_label: str) -> str:
        level = min(level + 1, 6)
        slug = self.slugify(text)
        plain = re.sub(r"`([^`]+)`", r"\1", text)
        if self.collect_toc:
            self.toc.append((level, slug, plain, doc_label))
        return (
            f'<h{level} id="{slug}">{self.inline(text)}'
            f'<a class="anchor" href="#{slug}">#</a></h{level}>'
        )

    def table(self, rows: list[str]) -> str:
        parsed = [self.split_table_row(row) for row in rows]
        if len(parsed) < 2:
            return ""
        header = parsed[0]
        body = parsed[2:]
        out = ['<div class="table-wrap"><table><thead><tr>']
        out.extend(f"<th>{self.inline(cell)}</th>" for cell in header)
        out.append("</tr></thead><tbody>")
        for row in body:
            out.append("<tr>")
            out.extend(f"<td>{self.inline(cell)}</td>" for cell in row)
            out.append("</tr>")
        out.append("</tbody></table></div>")
        return "".join(out)

    @staticmethod
    def split_table_row(row: str) -> list[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [cell.strip() for cell in row.split("|")]

    @staticmethod
    def is_table_separator(row: str) -> bool:
        cells = MarkdownRenderer.split_table_row(row)
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)

    @staticmethod
    def starts_block(line: str, next_line: str | None = None) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if stripped.startswith("```"):
            return True
        if re.match(r"#{1,6}\s+", stripped):
            return True
        if re.match(r"([-*])\s+", stripped):
            return True
        if re.match(r"\d+\.\s+", stripped):
            return True
        if stripped.startswith(">"):
            return True
        if stripped.startswith("|") and next_line and MarkdownRenderer.is_table_separator(next_line.strip()):
            return True
        return False

    def render(self, markdown_text: str, doc_label: str) -> str:
        lines = markdown_text.splitlines()
        out: list[str] = []
        paragraph: list[str] = []
        i = 0

        def flush_paragraph() -> None:
            nonlocal paragraph
            if paragraph:
                text = " ".join(part.strip() for part in paragraph).strip()
                if text:
                    out.append(f"<p>{self.inline(text)}</p>")
                paragraph = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            next_line = lines[i + 1] if i + 1 < len(lines) else None

            if not stripped:
                flush_paragraph()
                i += 1
                continue

            if stripped.startswith("```"):
                flush_paragraph()
                lang = stripped[3:].strip() or "text"
                i += 1
                code_lines: list[str] = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1
                code = html.escape("\n".join(code_lines))
                out.append(
                    f'<pre><div class="code-label">{html.escape(lang)}</div>'
                    f"<code>{code}</code></pre>"
                )
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                flush_paragraph()
                out.append(self.heading(len(heading_match.group(1)), heading_match.group(2), doc_label))
                i += 1
                continue

            if stripped.startswith("|") and next_line and self.is_table_separator(next_line.strip()):
                flush_paragraph()
                rows = [stripped, next_line.strip()]
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append(lines[i].strip())
                    i += 1
                out.append(self.table(rows))
                continue

            if stripped.startswith(">"):
                flush_paragraph()
                quote_lines: list[str] = []
                while i < len(lines) and lines[i].strip().startswith(">"):
                    quote_lines.append(lines[i].strip().lstrip(">").strip())
                    i += 1
                quote = " ".join(line for line in quote_lines if line)
                out.append(f"<blockquote><p>{self.inline(quote)}</p></blockquote>")
                continue

            unordered = re.match(r"^[-*]\s+(.+)$", stripped)
            ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
            if unordered or ordered:
                flush_paragraph()
                tag = "ul" if unordered else "ol"
                out.append(f"<{tag}>")
                while i < len(lines):
                    item_line = lines[i].strip()
                    item_match = re.match(r"^[-*]\s+(.+)$", item_line)
                    if tag == "ol":
                        item_match = re.match(r"^\d+\.\s+(.+)$", item_line)
                    if not item_match:
                        break
                    item_parts = [item_match.group(1)]
                    i += 1
                    while i < len(lines):
                        cont = lines[i].strip()
                        following = lines[i + 1] if i + 1 < len(lines) else None
                        if not cont:
                            break
                        if tag == "ul" and re.match(r"^[-*]\s+(.+)$", cont):
                            break
                        if tag == "ol" and re.match(r"^\d+\.\s+(.+)$", cont):
                            break
                        if self.starts_block(cont, following):
                            break
                        item_parts.append(cont)
                        i += 1
                    out.append(f"<li>{self.inline(' '.join(item_parts))}</li>")
                out.append(f"</{tag}>")
                continue

            paragraph.append(line)
            if next_line is None or self.starts_block(next_line, lines[i + 2] if i + 2 < len(lines) else None):
                flush_paragraph()
            i += 1

        flush_paragraph()
        return "\n".join(out)


def stable_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def section_card(doc: SourceDoc) -> str:
    rel = doc.path.relative_to(REPO_READING)
    return (
        '<div class="source-card">'
        f'<strong>{html.escape(doc.label)}</strong>'
        f'<span>{html.escape(doc.summary)}</span>'
        f'<code>{html.escape(str(rel))}</code>'
        "</div>"
    )


def external_card(doc: ExternalDoc) -> str:
    return (
        '<a class="source-card" '
        f'href="{html.escape(doc.url)}">'
        f'<strong>{html.escape(doc.label)}</strong>'
        f'<span>{html.escape(doc.summary)}</span>'
        f'<code>{html.escape(doc.url)}</code>'
        "</a>"
    )


def synthesis_section() -> str:
    return """
<section class="doc-section synthesis" id="synthesis">
  <div class="doc-kicker">Synthesis</div>
  <h2>Current MegaMoE Map<a class="anchor" href="#synthesis">#</a></h2>
  <p>This HTML is the human-readable layer. It intentionally does not mirror the
  markdown notes one-to-one: the first half is a curated reading path, while the
  second half keeps the source notes as expandable appendices.</p>

  <h3 id="one-screen-model">One-Screen Model<a class="anchor" href="#one-screen-model">#</a></h3>
  <pre><div class="code-label">text</div><code>routing/top-k already done outside kernel
  -> dispatch pull into L1 ring slots
  -> Linear1 GEMM
  -> activation/SwiGLU epilogue and FP8 requantization into L2 ring slots
  -> Linear2 GEMM
  -> remote write-back into token-owner combine slots
  -> barrier
  -> local top-k-slot reduction to y</code></pre>

  <div class="map-grid">
    <div class="map-card"><b>Scope</b><span>Public SM100 forward path. Treat
    training backward, SM90 variants, and cross-node RDMA claims as open until
    there is separate code evidence.</span></div>
    <div class="map-card"><b>Runtime protocol</b><span>Symmetric peer buffers,
    pool tokens, ring slots, counters, source metadata, and phase barriers are
    cross-stage state, not dispatch-only details.</span></div>
    <div class="map-card"><b>Scheduling</b><span>Wave is a higher-level work
    window. Ring block is a runtime buffer unit. GEMM has its own K-tile
    pipeline.</span></div>
    <div class="map-card"><b>Numerics</b><span>Input / weight quantization is
    outside the kernel. Linear1 activation produces FP8 L2 inputs. Combine
    reduces preweighted BF16 route outputs in FP32 registers.</span></div>
  </div>

  <h3 id="synthesis-reading-order">Reading Order<a class="anchor" href="#synthesis-reading-order">#</a></h3>
  <ol>
    <li>Start with the lowering map to keep algorithm, EP reference, and kernel
    rewrite separate.</li>
    <li>Read runtime protocol before dispatch so symmetric memory, pool/ring
    slots, counters, and source metadata have one home.</li>
    <li>Read scheduling before individual compute stages so wave, ring, and
    tile units do not get mixed together.</li>
    <li>Read dispatch, quantization, activation, GEMM, and combine in execution
    order after the protocol layer is clear.</li>
    <li>Use the hardware appendices only when a term blocks the source-level
    logic.</li>
  </ol>

  <h3 id="synthesis-stage-map">Stage Map<a class="anchor" href="#synthesis-stage-map">#</a></h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Stage</th><th>Main state</th><th>Producer-consumer unit</th><th>Key correctness edge</th></tr></thead>
    <tbody>
      <tr><td>Dispatch</td><td><code>l1_acts</code>, <code>l1_acts_sf</code>, route metadata</td><td>Ring block / pool token</td><td>Source metadata and L1 full counters publish route ownership.</td></tr>
      <tr><td>Linear1</td><td>FP32 TMEM accumulator</td><td>GEMM tile / K-tile pipeline</td><td>Block-scaled UMMA consumes payloads and scale metadata consistently.</td></tr>
      <tr><td>Activation</td><td><code>l2_acts</code>, <code>l2_acts_sf</code></td><td>TMEM accumulator tile to L2 ring slot</td><td>Gate/up BF16 rounding, SwiGLU, top-k weight, amax, and FP8 requantization happen before Linear2.</td></tr>
      <tr><td>Linear2</td><td>FP32 TMEM accumulator</td><td>GEMM tile / K-tile pipeline</td><td>Consumes only ready L2 ring slots.</td></tr>
      <tr><td>Write-back + combine</td><td><code>combine_token_buffer</code>, <code>y</code></td><td>Top-k slot / hidden chunk</td><td>Remote writes finish before local fixed-order top-k-slot reduction.</td></tr>
    </tbody>
  </table></div>

  <h3 id="synthesis-open-questions">Useful Open Questions<a class="anchor" href="#synthesis-open-questions">#</a></h3>
  <ul>
    <li>Whether a public backward / training MegaMoE path appears, and whether
    it keeps the same fusion boundary.</li>
    <li>How production code chooses capacity such as
    <code>num_max_tokens_per_rank</code> under real load imbalance.</li>
    <li>How far this NVLink-domain design generalizes beyond one NVLink /
    NVSwitch domain.</li>
    <li>Which details are Blackwell-specific hardware contracts rather than
    durable algorithmic ideas.</li>
  </ul>
</section>

<section class="doc-section" id="lowering-story">
  <div class="doc-kicker">Human Reading Path</div>
  <h2>Lowering Story<a class="anchor" href="#lowering-story">#</a></h2>
  <p>The easiest way to read MegaMoE is to keep four layers separate. Each layer
  preserves the same MoE semantics but changes the execution representation.</p>

  <div class="table-wrap"><table>
    <thead><tr><th>Layer</th><th>Question</th><th>What Changes</th><th>Do Not Confuse With</th></tr></thead>
    <tbody>
      <tr><td>Algorithm</td><td>What is MoE computing?</td><td>Top-k expert FFNs and weighted sum.</td><td>Rank ownership or GPU scheduling.</td></tr>
      <tr><td>Distributed EP reference</td><td>Which rank owns each route?</td><td>Dispatch routes to expert owners, compute locally, combine back.</td><td>One fused kernel implementation.</td></tr>
      <tr><td>MegaMoE execution rewrite</td><td>How is EP implemented without materialized all-to-all stages?</td><td>Symmetric buffers, ring slots, persistent workers, overlapped communication and compute.</td><td>Mathematical model changes.</td></tr>
      <tr><td>CUDA / hardware mapping</td><td>Which GPU mechanisms carry the rewrite?</td><td>TMA, UMMA, TMEM, shared memory, NVLink peer stores, barriers, counters.</td><td>High-level wave policy.</td></tr>
    </tbody>
  </table></div>

  <p>A useful reading discipline: when a variable appears, first ask which layer
  it belongs to. For example, <code>topk_idx</code> is model-routing state,
  <code>TokenSrcMetadata</code> is EP ownership state, <code>ring_block_idx</code>
  is fused-kernel buffer state, and <code>accum_stage_idx</code> is GEMM/TMEM
  pipeline state.</p>
</section>

<section class="doc-section" id="execution-timeline">
  <div class="doc-kicker">Execution Timeline</div>
  <h2>Forward Path Timeline<a class="anchor" href="#execution-timeline">#</a></h2>
  <p>The public code path is best read as a set of producer-consumer edges rather
  than as a flat list of kernels. Most confusion came from mixing the units:
  wave, ring block, GEMM tile, K-tile stage, and top-k slot are different units.</p>

  <div class="table-wrap"><table>
    <thead><tr><th>Step</th><th>Producer</th><th>Consumer</th><th>Memory / State</th><th>Scheduling Unit</th></tr></thead>
    <tbody>
      <tr><td>Dispatch pull</td><td>Source token-owner rank</td><td>Expert-owner rank</td><td><code>x</code>, <code>x_sf</code>, <code>topk_idx</code>, <code>topk_weights</code> into <code>l1_acts</code> ring slots</td><td>Pool token / ring block</td></tr>
      <tr><td>Linear1</td><td>L1 ring slot</td><td>Activation epilogue</td><td>FP32 accumulator in TMEM</td><td>GEMM tile plus K-tile stages</td></tr>
      <tr><td>SwiGLU epilogue</td><td>TMEM accumulator</td><td>Linear2</td><td><code>l2_acts</code>, <code>l2_acts_sf</code> in global ring buffer</td><td>Accumulator tile / ring block</td></tr>
      <tr><td>Linear2</td><td>L2 ring slot</td><td>Write-back epilogue</td><td>FP32 accumulator in TMEM</td><td>GEMM tile plus K-tile stages</td></tr>
      <tr><td>Remote write-back</td><td>Expert-owner rank</td><td>Token-owner rank</td><td><code>combine_token_buffer[topk_slot, token]</code></td><td>Route output / top-k slot</td></tr>
      <tr><td>Final combine</td><td>Local combine slots</td><td>Output tensor</td><td>FP32 register accumulation, BF16 <code>y</code></td><td>Token / hidden chunk</td></tr>
    </tbody>
  </table></div>

  <div class="callout">The final combine is not a remote pull. Linear2 epilogue
  pushes route outputs to the token-owner rank's combine buffer, then a
  phase-level NVLink barrier makes those writes visible before local reduction.</div>
</section>

<section class="doc-section" id="concept-crosswalk">
  <div class="doc-kicker">Concept Crosswalk</div>
  <h2>Terms That Should Not Collapse Together<a class="anchor" href="#concept-crosswalk">#</a></h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Term</th><th>Layer</th><th>Short Meaning</th><th>Typical Misread</th></tr></thead>
    <tbody>
      <tr><td>Wave</td><td>Execution scheduling</td><td>A high-level window over a subset of expert work.</td><td>Not the same as a ring block or CUDA warp.</td></tr>
      <tr><td>Ring block</td><td>Buffer protocol</td><td>A reusable producer-consumer slot generation.</td><td>Not a full expert or a whole wave.</td></tr>
      <tr><td>GEMM tile</td><td>Compute tiling</td><td>A block of C produced by UMMA into TMEM.</td><td>Not the same as the communication chunk.</td></tr>
      <tr><td>K-tile stage</td><td>GEMM pipeline</td><td>Shared-memory pipeline stage for reduction dimension tiles.</td><td>Not the MoE pipeline stage.</td></tr>
      <tr><td>Top-k slot</td><td>Model route / combine</td><td>The slot position of a token's selected expert route.</td><td>Not necessarily sorted expert ID order.</td></tr>
      <tr><td>TMEM</td><td>Blackwell tensor-core path</td><td>Tensor memory holding accumulator tiles near UMMA / epilogue.</td><td>Not TMA and not global memory.</td></tr>
      <tr><td>TMA</td><td>Memory movement engine</td><td>Asynchronous tensor memory access between global and shared memory.</td><td>Not a compute unit and not TMEM.</td></tr>
    </tbody>
  </table></div>
</section>

<section class="doc-section" id="evidence-and-gaps">
  <div class="doc-kicker">Evidence</div>
  <h2>What Is Solid Versus Still Open<a class="anchor" href="#evidence-and-gaps">#</a></h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Status</th><th>Conclusion</th><th>Reason</th></tr></thead>
    <tbody>
      <tr><td>Code-backed</td><td>Current public MegaMoE path is a forward kernel path.</td><td>Notes are anchored to DeepGEMM public SM100 FP8/FP4 files and tests.</td></tr>
      <tr><td>Code-backed</td><td>Input and weight quantization happen before the MegaMoE kernel boundary.</td><td>The wrapper/test path passes quantized payloads and scale factors into the kernel ABI.</td></tr>
      <tr><td>Code-backed</td><td>Activation epilogue applies top-k weight before Linear2, so final combine only sums slots.</td><td>Linear1 epilogue writes preweighted <code>l2_acts</code>; combine does not reread <code>topk_weights</code>.</td></tr>
      <tr><td>Code-backed</td><td>Combine write-back is remote push, followed by a phase-level barrier and local reduction.</td><td>Linear2 epilogue uses <code>sym_buffer.map(dst_ptr, dst_rank_idx)</code>; final combine iterates local tokens.</td></tr>
      <tr><td>Inference</td><td>The public path is best understood as NVLink-domain peer memory, not RDMA all-to-all.</td><td>Kernel comments, barriers, and benchmark accounting all point to NVLink-style peer access; full backend audit remains separate.</td></tr>
      <tr><td>Open</td><td>Backward/training MegaMoE kernel support.</td><td>No public backward path has been pinned in these notes yet.</td></tr>
    </tbody>
  </table></div>
</section>
"""


def visuals_section() -> str:
    return """
<section class="doc-section" id="visual-models">
  <div class="doc-kicker">Visual Mental Models</div>
  <h2>Diagrams For Human Reading<a class="anchor" href="#visual-models">#</a></h2>
  <p>These are original explanatory diagrams, not copied vendor or blog images.
  They are intentionally simplified: the goal is to anchor the mental model
  before reading the detailed code notes.</p>

  <div class="ref-grid">
    <a class="ref-card" href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">
      <strong>Official Hopper diagrams</strong>
      <span>Real full-chip / SM block diagrams and TMA context.</span>
    </a>
    <a class="ref-card" href="https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html">
      <strong>Hopper tuning guide</strong>
      <span>TMA, occupancy, shared memory, L2, NVLink, and clusters.</span>
    </a>
    <a class="ref-card" href="https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html">
      <strong>Blackwell tuning guide</strong>
      <span>Blackwell SM, memory-system, cluster, and NVLink details.</span>
    </a>
  </div>

  <div class="diagram-panel" id="diagram-moe-ep">
    <h3>MoE Semantics To Distributed EP<a class="anchor" href="#diagram-moe-ep">#</a></h3>
    <svg class="diagram-svg" viewBox="0 0 1040 480" role="img" aria-label="MoE routing and expert parallel reference flow">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#365f91"></path>
        </marker>
      </defs>
      <rect class="svg-band" x="24" y="34" width="992" height="156"></rect>
      <text class="svg-kicker" x="44" y="60">Algorithm layer</text>
      <rect class="svg-box accent" x="52" y="88" width="150" height="70"></rect>
      <text class="svg-title" x="127" y="116">tokens</text>
      <text class="svg-small" x="127" y="140">hidden states</text>
      <rect class="svg-box" x="288" y="88" width="150" height="70"></rect>
      <text class="svg-title" x="363" y="116">router</text>
      <text class="svg-small" x="363" y="140">top-k experts</text>
      <rect class="svg-box" x="524" y="72" width="180" height="104"></rect>
      <text class="svg-title" x="614" y="104">experts</text>
      <text class="svg-small" x="614" y="128">FFN per route</text>
      <text class="svg-small" x="614" y="150">gate / up / down</text>
      <rect class="svg-box accent" x="790" y="88" width="150" height="70"></rect>
      <text class="svg-title" x="865" y="116">combine</text>
      <text class="svg-small" x="865" y="140">weighted sum</text>
      <path class="svg-arrow" marker-end="url(#arrow)" d="M202,123 L286,123"></path>
      <path class="svg-arrow" marker-end="url(#arrow)" d="M438,123 L522,123"></path>
      <path class="svg-arrow" marker-end="url(#arrow)" d="M704,123 L788,123"></path>

      <rect class="svg-band" x="24" y="246" width="992" height="190"></rect>
      <text class="svg-kicker" x="44" y="272">Distributed EP reference layer</text>
      <rect class="svg-box accent2" x="56" y="308" width="150" height="78"></rect>
      <text class="svg-title" x="131" y="338">owner rank</text>
      <text class="svg-small" x="131" y="362">token batch</text>
      <rect class="svg-box" x="290" y="286" width="172" height="124"></rect>
      <text class="svg-title" x="376" y="318">dispatch</text>
      <text class="svg-small" x="376" y="342">route tokens</text>
      <text class="svg-small" x="376" y="364">to expert owners</text>
      <rect class="svg-box" x="544" y="286" width="190" height="124"></rect>
      <text class="svg-title" x="639" y="318">local experts</text>
      <text class="svg-small" x="639" y="342">grouped compute</text>
      <text class="svg-small" x="639" y="364">per rank</text>
      <rect class="svg-box accent2" x="816" y="308" width="150" height="78"></rect>
      <text class="svg-title" x="891" y="338">combine back</text>
      <text class="svg-small" x="891" y="362">token owner</text>
      <path class="svg-arrow" marker-end="url(#arrow)" d="M206,347 L288,347"></path>
      <path class="svg-arrow" marker-end="url(#arrow)" d="M462,347 L542,347"></path>
      <path class="svg-arrow" marker-end="url(#arrow)" d="M734,347 L814,347"></path>
    </svg>
    <p class="diagram-caption">MegaMoE keeps the MoE semantics but rewrites the EP
    implementation boundary: dispatch and combine become device-side peer-memory
    protocols inside one persistent forward kernel.</p>
  </div>

  <div class="diagram-panel" id="diagram-megamoe-pipeline">
    <h3>MegaMoE Forward Pipeline<a class="anchor" href="#diagram-megamoe-pipeline">#</a></h3>
    <svg class="diagram-svg" viewBox="0 0 1120 540" role="img" aria-label="MegaMoE fused forward pipeline">
      <defs>
        <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#365f91"></path>
        </marker>
      </defs>
      <rect class="svg-lane" x="32" y="54" width="1056" height="110"></rect>
      <text class="svg-kicker" x="52" y="82">Communication / ring-buffer lane</text>
      <rect class="svg-box accent" x="70" y="104" width="146" height="44"></rect>
      <text class="svg-title" x="143" y="132">sym input</text>
      <rect class="svg-box" x="292" y="96" width="164" height="60"></rect>
      <text class="svg-title" x="374" y="122">dispatch pull</text>
      <text class="svg-small" x="374" y="144">L1 ring slots</text>
      <rect class="svg-box" x="646" y="96" width="164" height="60"></rect>
      <text class="svg-title" x="728" y="122">L2 ring ready</text>
      <text class="svg-small" x="728" y="144">l2_full_count</text>
      <rect class="svg-box accent" x="884" y="96" width="164" height="60"></rect>
      <text class="svg-title" x="966" y="122">remote push</text>
      <text class="svg-small" x="966" y="144">combine slots</text>

      <rect class="svg-lane" x="32" y="210" width="1056" height="122"></rect>
      <text class="svg-kicker" x="52" y="238">Tensor-core GEMM lane</text>
      <rect class="svg-box accent2" x="292" y="260" width="164" height="54"></rect>
      <text class="svg-title" x="374" y="284">Linear1</text>
      <text class="svg-small" x="374" y="304">UMMA -> TMEM</text>
      <rect class="svg-box accent2" x="646" y="260" width="164" height="54"></rect>
      <text class="svg-title" x="728" y="284">Linear2</text>
      <text class="svg-small" x="728" y="304">UMMA -> TMEM</text>

      <rect class="svg-lane" x="32" y="378" width="1056" height="104"></rect>
      <text class="svg-kicker" x="52" y="406">Epilogue / scalar lane</text>
      <rect class="svg-box" x="470" y="424" width="144" height="44"></rect>
      <text class="svg-title" x="542" y="452">SwiGLU</text>
      <rect class="svg-box" x="884" y="424" width="164" height="44"></rect>
      <text class="svg-title" x="966" y="452">local reduce</text>

      <path class="svg-arrow" marker-end="url(#arrow2)" d="M216,126 L290,126"></path>
      <path class="svg-arrow" marker-end="url(#arrow2)" d="M456,126 C510,126 510,287 290,287"></path>
      <path class="svg-arrow" marker-end="url(#arrow2)" d="M456,287 L468,446"></path>
      <path class="svg-arrow" marker-end="url(#arrow2)" d="M614,446 C646,446 620,126 644,126"></path>
      <path class="svg-arrow" marker-end="url(#arrow2)" d="M810,126 C850,126 850,287 810,287"></path>
      <path class="svg-arrow" marker-end="url(#arrow2)" d="M810,287 L882,126"></path>
      <path class="svg-arrow" marker-end="url(#arrow2)" d="M966,156 L966,422"></path>
      <line class="svg-dash" x1="852" y1="366" x2="1072" y2="366"></line>
      <text class="svg-small left" x="860" y="356">phase-level barrier before final combine</text>
    </svg>
    <p class="diagram-caption">The key distinction is granularity: MoE stages are
    the conceptual pipeline, ring blocks are reusable buffer slots, and GEMM has
    a separate K-tile pipeline inside Linear1 / Linear2.</p>
  </div>

  <div class="diagram-panel" id="diagram-symmetric-peer">
    <h3>Symmetric Memory Peer Path<a class="anchor" href="#diagram-symmetric-peer">#</a></h3>
    <svg class="diagram-svg" viewBox="0 0 1120 520" role="img" aria-label="Symmetric memory peer address path for dispatch pull and combine push">
      <defs>
        <marker id="arrow-peer" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#365f91"></path>
        </marker>
      </defs>
      <rect class="svg-band" x="40" y="54" width="1040" height="126"></rect>
      <text class="svg-kicker" x="62" y="82">Setup before fused kernel</text>
      <rect class="svg-box accent" x="92" y="110" width="190" height="48"></rect>
      <text class="svg-title" x="187" y="139">rank-local buffer</text>
      <rect class="svg-box accent" x="362" y="110" width="190" height="48"></rect>
      <text class="svg-title" x="457" y="139">same layout</text>
      <rect class="svg-box accent" x="632" y="110" width="190" height="48"></rect>
      <text class="svg-title" x="727" y="139">rendezvous</text>
      <rect class="svg-box accent" x="902" y="110" width="130" height="48"></rect>
      <text class="svg-title" x="967" y="139">peer ptrs</text>
      <path class="svg-arrow" marker-end="url(#arrow-peer)" d="M282,134 L360,134"></path>
      <path class="svg-arrow" marker-end="url(#arrow-peer)" d="M552,134 L630,134"></path>
      <path class="svg-arrow" marker-end="url(#arrow-peer)" d="M822,134 L900,134"></path>

      <rect class="svg-band" x="40" y="226" width="1040" height="218"></rect>
      <text class="svg-kicker" x="62" y="254">Inside MegaMoE persistent kernel</text>
      <rect class="svg-box" x="92" y="290" width="210" height="74"></rect>
      <text class="svg-title" x="197" y="318">source rank</text>
      <text class="svg-small" x="197" y="342">x / x_sf / topk</text>
      <rect class="svg-box accent2" x="455" y="276" width="210" height="102"></rect>
      <text class="svg-title" x="560" y="308">expert rank</text>
      <text class="svg-small" x="560" y="332">dispatch pull</text>
      <text class="svg-small" x="560" y="354">compute Linear1/2</text>
      <rect class="svg-box" x="818" y="290" width="210" height="74"></rect>
      <text class="svg-title" x="923" y="318">token owner</text>
      <text class="svg-small" x="923" y="342">combine slots / y</text>
      <path class="svg-arrow" marker-end="url(#arrow-peer)" d="M454,322 C390,292 372,292 304,322"></path>
      <text class="svg-small left" x="326" y="288">pull remote input</text>
      <path class="svg-arrow" marker-end="url(#arrow-peer)" d="M666,336 C724,372 758,372 816,336"></path>
      <text class="svg-small left" x="692" y="382">push route output</text>
      <line class="svg-dash" x1="760" y1="410" x2="1012" y2="410"></line>
      <text class="svg-small left" x="768" y="432">barrier before local combine reduction</text>
    </svg>
    <p class="diagram-caption">The symmetric-memory setup gives the kernel a
    peer pointer table. In dispatch, the expert rank reads remote source-token
    payloads into local L1 ring slots. In Linear2 write-back, the expert rank
    writes route outputs into the token-owner rank's combine slots.</p>
  </div>

  <div class="diagram-panel" id="diagram-gpu-memory">
    <h3>GPU Hardware / Memory Map<a class="anchor" href="#diagram-gpu-memory">#</a></h3>
    <div class="inline-ref">
      For real hardware block diagrams, compare this simplified map with
      <a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">NVIDIA's Hopper full-chip and SM diagrams</a>.
      For Blackwell-specific constraints, use the
      <a href="https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html">Blackwell tuning guide</a>.
    </div>
    <svg class="diagram-svg" viewBox="0 0 1040 560" role="img" aria-label="GPU memory hierarchy and SM components relevant to MegaMoE">
      <defs>
        <marker id="arrow3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#365f91"></path>
        </marker>
      </defs>
      <rect class="svg-box wide accent" x="78" y="52" width="884" height="62"></rect>
      <text class="svg-title" x="520" y="82">HBM / symmetric global memory</text>
      <text class="svg-small" x="520" y="104">persistent tensors, ring buffers, peer-visible combine buffers</text>
      <rect class="svg-box wide" x="132" y="154" width="776" height="58"></rect>
      <text class="svg-title" x="520" y="184">L2 cache</text>
      <text class="svg-small" x="520" y="204">shared by SMs, performance layer not explicit storage contract</text>
      <rect class="svg-box sm" x="112" y="260" width="816" height="220"></rect>
      <text class="svg-kicker" x="138" y="290">one SM mental model</text>
      <rect class="svg-box" x="150" y="322" width="158" height="64"></rect>
      <text class="svg-title" x="229" y="350">registers</text>
      <text class="svg-small" x="229" y="372">per thread</text>
      <rect class="svg-box" x="336" y="322" width="168" height="64"></rect>
      <text class="svg-title" x="420" y="350">shared / L1</text>
      <text class="svg-small" x="420" y="372">CTA-visible staging</text>
      <rect class="svg-box accent2" x="532" y="322" width="158" height="64"></rect>
      <text class="svg-title" x="611" y="350">TMEM</text>
      <text class="svg-small" x="611" y="372">accumulator tiles</text>
      <rect class="svg-box" x="718" y="322" width="168" height="64"></rect>
      <text class="svg-title" x="802" y="350">TMA</text>
      <text class="svg-small" x="802" y="372">global <-> shared</text>
      <rect class="svg-box accent2" x="236" y="416" width="192" height="44"></rect>
      <text class="svg-title" x="332" y="444">Tensor Cores / UMMA</text>
      <rect class="svg-box" x="480" y="416" width="172" height="44"></rect>
      <text class="svg-title" x="566" y="444">CUDA cores</text>
      <path class="svg-arrow" marker-end="url(#arrow3)" d="M520,114 L520,152"></path>
      <path class="svg-arrow" marker-end="url(#arrow3)" d="M520,212 L520,258"></path>
      <path class="svg-arrow" marker-end="url(#arrow3)" d="M802,322 C802,252 520,252 520,214"></path>
      <path class="svg-arrow" marker-end="url(#arrow3)" d="M504,354 L530,354"></path>
      <path class="svg-arrow" marker-end="url(#arrow3)" d="M611,386 L428,416"></path>
      <path class="svg-arrow" marker-end="url(#arrow3)" d="M611,386 L566,416"></path>
      <text class="svg-small left" x="102" y="520">For MegaMoE: TMA moves payloads / scale factors into shared memory; UMMA accumulates C tiles in TMEM.</text>
      <text class="svg-small left" x="102" y="542">Epilogues move from TMEM through shared/registers to ring buffers or combine buffers.</text>
    </svg>
    <p class="diagram-caption">This is not a vendor-accurate floorplan. It is a
    reading map for the memory and execution concepts that appear in the kernel
    notes.</p>
  </div>

  <div class="diagram-panel" id="diagram-unit-crosswalk">
    <h3>Scheduling Unit Crosswalk<a class="anchor" href="#diagram-unit-crosswalk">#</a></h3>
    <svg class="diagram-svg" viewBox="0 0 1040 430" role="img" aria-label="Different scheduling and buffer units in MegaMoE">
      <rect class="svg-box wide accent" x="86" y="52" width="868" height="54"></rect>
      <text class="svg-title" x="520" y="84">wave: high-level expert-work window</text>
      <rect class="svg-box wide" x="142" y="130" width="756" height="54"></rect>
      <text class="svg-title" x="520" y="162">pool block: logical token range for an expert</text>
      <rect class="svg-box wide" x="198" y="208" width="644" height="54"></rect>
      <text class="svg-title" x="520" y="240">ring block: reusable producer-consumer slot generation</text>
      <rect class="svg-box wide accent2" x="254" y="286" width="532" height="54"></rect>
      <text class="svg-title" x="520" y="318">GEMM tile: M x N block, with K-tile pipeline</text>
      <rect class="svg-box wide" x="310" y="364" width="420" height="44"></rect>
      <text class="svg-title" x="520" y="392">top-k slot: final route-output combine index</text>
      <text class="svg-small left" x="88" y="122">higher-level scheduling</text>
      <text class="svg-small left" x="198" y="280">buffer protocol</text>
      <text class="svg-small left" x="254" y="356">compute tiling</text>
    </svg>
    <p class="diagram-caption">When reading source, always identify the unit
    first. A barrier or counter that is correct for a ring block is usually not
    a wave barrier, and a GEMM K-stage is not a MoE pipeline stage.</p>
  </div>

  <h3 id="visual-references">Reference Material Used For Visual Vocabulary<a class="anchor" href="#visual-references">#</a></h3>
  <ul>
    <li>NVIDIA Hopper architecture overview and SM / full-chip diagrams:
    <a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">NVIDIA Hopper Architecture In-Depth</a>.</li>
    <li>CUDA architecture details for Hopper and Blackwell:
    <a href="https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html">Hopper Tuning Guide</a> and
    <a href="https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html">Blackwell Tuning Guide</a>.</li>
    <li>MoE / MegaMoE stage semantics come from the local DeepSeek-V4 paper copy
    and the code-backed source notes rendered below.</li>
  </ul>
</section>
"""


def source_notes_intro() -> str:
    return """
<section class="doc-section" id="source-notes">
  <div class="doc-kicker">Appendix</div>
  <h2>Source Notes<a class="anchor" href="#source-notes">#</a></h2>
  <p>The sections below are generated from the markdown notes and kept
  collapsible on purpose. Use them when the curated story above is not enough
  and you need code anchors, tensor names, or detailed caveats. Reusable CUDA /
  GPU hardware notes live in the separate
  <a href="https://zyeric.github.io/gpu-hardware-notes/">GPU Hardware Notes</a>
  site instead of being vendored into this MegaMoE case study.</p>
</section>
"""


def render_doc(renderer: MarkdownRenderer, doc: SourceDoc) -> str:
    text = doc.path.read_text(encoding="utf-8")
    rel = doc.path.relative_to(REPO_READING)
    body = renderer.render(text, doc.label)
    group_class = stable_id(doc.group)
    return (
        f'<details class="doc-section source-detail {group_class}" id="source-{stable_id(doc.label)}">\n'
        f'  <summary><span>{html.escape(doc.group)} / {html.escape(doc.label)}</span>'
        f'<code>{html.escape(str(rel))}</code></summary>\n'
        f'<div class="source-body">{body}</div>\n'
        "</details>"
    )


def toc_html() -> str:
    parts = [
        '<a class="top" href="#synthesis">Current MegaMoE Map</a>',
        '<a class="sub" href="#one-screen-model">One-Screen Model</a>',
        '<a class="sub" href="#synthesis-reading-order">Reading Order</a>',
        '<a class="sub" href="#synthesis-stage-map">Stage Map</a>',
        '<a class="top" href="#visual-models">Visual Mental Models</a>',
        '<a class="sub" href="#diagram-moe-ep">MoE To EP</a>',
        '<a class="sub" href="#diagram-megamoe-pipeline">MegaMoE Pipeline</a>',
        '<a class="sub" href="#diagram-symmetric-peer">Symmetric Peer Path</a>',
        '<a class="sub" href="#diagram-gpu-memory">GPU / Memory Map</a>',
        '<a class="sub" href="#diagram-unit-crosswalk">Unit Crosswalk</a>',
        '<a class="top" href="#lowering-story">Lowering Story</a>',
        '<a class="top" href="#execution-timeline">Forward Path Timeline</a>',
        '<a class="top" href="#concept-crosswalk">Concept Crosswalk</a>',
        '<a class="top" href="#evidence-and-gaps">Evidence And Gaps</a>',
        '<a class="top" href="#source-notes">Source Notes</a>',
        '<span class="nav-label">Guide Notes</span>',
    ]
    for doc in GUIDE_DOCS:
        parts.append(f'<a class="sub" href="#source-{stable_id(doc.label)}">{html.escape(doc.label)}</a>')
    parts.extend([
        '<span class="nav-label">Core Notes</span>',
    ])
    for doc in CORE_DOCS:
        parts.append(f'<a class="sub" href="#source-{stable_id(doc.label)}">{html.escape(doc.label)}</a>')
    parts.append('<span class="nav-label">External Hardware Notes</span>')
    for doc in HARDWARE_DOCS:
        parts.append(f'<a class="sub" href="{html.escape(doc.url)}">{html.escape(doc.label)}</a>')
    return "\n".join(parts)


def html_page(nav: str, body: str, generated: str) -> str:
    guide_cards = "\n".join(section_card(doc) for doc in GUIDE_DOCS)
    core_cards = "\n".join(section_card(doc) for doc in CORE_DOCS)
    hardware_cards = "\n".join(external_card(doc) for doc in HARDWARE_DOCS)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepSeek-V4 MegaMoE Reading Notes</title>
<style>
:root {{ color-scheme: light; --bg: #f4f5f3; --paper: #ffffff; --ink: #17191d; --muted: #646b75; --line: #d9dee2; --soft: #edf0f2; --accent: #006b5f; --accent2: #8a4a15; --accent3: #365f91; --code-bg: #101419; --code-ink: #e8eef6; --radius: 8px; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); line-height: 1.62; font-size: 16px; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.layout {{ display: grid; grid-template-columns: minmax(250px, 330px) minmax(0, 1fr); min-height: 100vh; }}
aside {{ position: sticky; top: 0; height: 100vh; overflow: auto; padding: 22px 18px; border-right: 1px solid var(--line); background: #fbfcfd; }}
.brand {{ font-weight: 760; line-height: 1.2; margin-bottom: 8px; }}
.meta {{ color: var(--muted); font-size: 12px; margin-bottom: 18px; }}
.nav a {{ display: block; color: #303640; padding: 4px 0; font-size: 13px; overflow-wrap: anywhere; }}
.nav a.sub {{ padding-left: 12px; color: #5c6470; }}
.nav a.minor {{ padding-left: 24px; color: #78808b; font-size: 12px; }}
.nav .nav-label {{ display: block; margin: 10px 0 3px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
main {{ max-width: 1160px; width: 100%; padding: 34px 44px 80px; }}
.hero, .doc-section {{ background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); }}
.hero {{ padding: 30px; margin-bottom: 18px; }}
.doc-section {{ padding: 28px 30px; margin: 18px 0; }}
.doc-kicker {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }}
h1 {{ font-size: 34px; line-height: 1.12; margin: 0 0 12px; letter-spacing: 0; }}
h2 {{ font-size: 25px; margin: 10px 0 16px; line-height: 1.25; letter-spacing: 0; }}
h3 {{ font-size: 20px; margin: 26px 0 10px; line-height: 1.3; letter-spacing: 0; border-top: 1px solid var(--soft); padding-top: 18px; }}
h4 {{ font-size: 17px; margin: 20px 0 8px; letter-spacing: 0; }}
h5, h6 {{ font-size: 15px; margin: 16px 0 8px; letter-spacing: 0; }}
.subtitle {{ color: var(--muted); max-width: 860px; margin: 0; }}
.summary-grid, .source-grid, .map-grid {{ display: grid; gap: 12px; margin-top: 22px; }}
.summary-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.source-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.map-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.summary-card, .source-card, .map-card {{ border: 1px solid var(--line); border-radius: var(--radius); background: #fafbfc; padding: 14px; }}
.summary-card strong, .source-card strong, .map-card b {{ display: block; margin-bottom: 6px; }}
.summary-card span, .source-card span, .map-card span {{ display: block; color: var(--muted); font-size: 13px; }}
.source-card code {{ display: block; margin-top: 8px; overflow-wrap: anywhere; }}
.flow {{ margin-top: 22px; display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; align-items: stretch; }}
.flow-step {{ border: 1px solid #cad5da; background: #f5faf8; border-radius: var(--radius); padding: 12px; min-height: 84px; }}
.flow-step b {{ display: block; color: #0e4f48; margin-bottom: 4px; }}
.flow-step small {{ color: var(--muted); }}
.callout {{ border-left: 4px solid var(--accent2); background: #fff8ef; padding: 12px 14px; border-radius: 0 var(--radius) var(--radius) 0; margin: 16px 0; }}
.ref-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0 22px; }}
.ref-card {{ display: block; border: 1px solid #bfd4d0; border-radius: var(--radius); padding: 14px; background: #f5fbfa; color: var(--ink); }}
.ref-card:hover {{ text-decoration: none; border-color: var(--accent); }}
.ref-card strong {{ display: block; color: #0e4f48; margin-bottom: 5px; }}
.ref-card span {{ display: block; color: var(--muted); font-size: 13px; }}
.inline-ref {{ border-left: 4px solid var(--accent3); background: #f5f8fc; padding: 10px 12px; margin: 10px 0 14px; color: #303640; border-radius: 0 var(--radius) var(--radius) 0; font-size: 14px; }}
.diagram-panel {{ border: 1px solid var(--line); border-radius: var(--radius); padding: 18px; margin: 18px 0; background: #fbfcfd; }}
.diagram-panel h3 {{ margin-top: 0; border-top: 0; padding-top: 0; }}
.diagram-svg {{ display: block; width: 100%; height: auto; background: #ffffff; border: 1px solid var(--line); border-radius: var(--radius); }}
.diagram-caption {{ color: var(--muted); font-size: 14px; margin: 10px 2px 0; }}
.svg-band, .svg-lane {{ fill: #f4f7f8; stroke: #d6dde2; stroke-width: 1.2; rx: 8; }}
.svg-lane {{ fill: #f7f8fa; }}
.svg-box {{ fill: #ffffff; stroke: #aeb8c2; stroke-width: 1.4; rx: 8; }}
.svg-box.accent {{ fill: #f0faf7; stroke: #0f766e; }}
.svg-box.accent2 {{ fill: #fff6ea; stroke: #9a5a1f; }}
.svg-box.wide {{ rx: 8; }}
.svg-box.sm {{ fill: #fcfcfd; stroke: #8b98a7; stroke-width: 1.6; rx: 8; }}
.svg-arrow {{ fill: none; stroke: #365f91; stroke-width: 2.4; }}
.svg-dash {{ stroke: #8a4a15; stroke-width: 2; stroke-dasharray: 7 6; }}
.svg-title {{ fill: #17191d; font-size: 18px; font-weight: 720; text-anchor: middle; dominant-baseline: middle; }}
.svg-small {{ fill: #646b75; font-size: 14px; text-anchor: middle; dominant-baseline: middle; }}
.svg-small.left {{ text-anchor: start; }}
.svg-kicker {{ fill: #646b75; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }}
.source-detail {{ padding: 0; overflow: hidden; }}
.source-detail summary {{ cursor: pointer; display: grid; grid-template-columns: auto 1fr; column-gap: 10px; row-gap: 4px; padding: 18px 22px; background: #fbfcfd; }}
.source-detail summary::marker {{ content: ""; }}
.source-detail summary::-webkit-details-marker {{ display: none; }}
.source-detail summary::before {{ content: "+"; color: var(--accent); font-weight: 800; grid-row: 1 / span 2; align-self: center; }}
.source-detail[open] summary {{ border-bottom: 1px solid var(--line); }}
.source-detail[open] summary::before {{ content: "-"; }}
.source-detail summary span {{ font-weight: 720; grid-column: 2; }}
.source-detail summary code {{ width: fit-content; color: var(--muted); grid-column: 2; }}
.source-body {{ padding: 24px 30px 30px; }}
.anchor {{ color: #a0a8b2; font-size: .75em; margin-left: 8px; opacity: 0; }}
h2:hover .anchor, h3:hover .anchor, h4:hover .anchor, h5:hover .anchor, h6:hover .anchor {{ opacity: 1; }}
p {{ margin: 10px 0; }}
ul, ol {{ padding-left: 22px; }}
li {{ margin: 5px 0; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; background: #edf1f5; padding: 1px 4px; border-radius: 4px; }}
pre {{ position: relative; overflow-x: auto; background: var(--code-bg); color: var(--code-ink); border-radius: var(--radius); padding: 34px 16px 16px; margin: 14px 0; border: 1px solid #252b33; }}
pre code {{ background: transparent; color: inherit; padding: 0; font-size: 13px; line-height: 1.55; }}
.code-label {{ position: absolute; top: 8px; left: 12px; font-size: 11px; color: #9ba6b5; text-transform: uppercase; }}
.table-wrap {{ overflow-x: auto; margin: 14px 0; border: 1px solid var(--line); border-radius: var(--radius); }}
table {{ border-collapse: collapse; width: 100%; min-width: 560px; font-size: 14px; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ background: #eef3f4; text-align: left; font-weight: 700; }}
blockquote {{ border-left: 4px solid var(--accent); background: #f5fbfa; margin: 14px 0; padding: 8px 16px; color: #2c3b3a; }}
@media (max-width: 1080px) {{ .layout {{ display: block; }} aside {{ position: relative; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }} main {{ padding: 22px 16px 60px; }} .summary-grid, .source-grid, .map-grid, .flow, .ref-grid {{ grid-template-columns: 1fr; }} .hero, .doc-section {{ padding: 20px; }} h1 {{ font-size: 28px; }} }}
@media print {{ aside {{ display: none; }} .layout {{ display: block; }} main {{ max-width: none; padding: 0; }} .hero, .doc-section {{ break-inside: avoid; border: 0; }} body {{ background: white; }} }}
</style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand">DeepSeek-V4 MegaMoE<br>Reading Notes</div>
    <div class="meta">Generated {html.escape(generated)}<br>Source: repo_reading Markdown</div>
    <nav class="nav">{nav}</nav>
  </aside>
  <main>
    <section class="hero" id="top">
      <h1>DeepSeek-V4 MegaMoE: Forward Path Reading Snapshot</h1>
      <p class="subtitle">A static, mobile-friendly reading page built from the
      repo notes. It is organized around the current code-backed understanding
      of the public DeepGEMM SM100 FP8/FP4 MegaMoE forward path.</p>
      <div class="summary-grid">
        <div class="summary-card"><strong>Lowering</strong><span>Math MoE -> distributed EP -> fused kernel execution -> CUDA/ISA mapping.</span></div>
        <div class="summary-card"><strong>Pipeline</strong><span>Dispatch, Linear1, activation, Linear2, write-back, combine.</span></div>
        <div class="summary-card"><strong>Memory</strong><span>Symmetric buffer, L1/L2 ring buffers, TMEM accumulators, shared staging.</span></div>
        <div class="summary-card"><strong>Evidence</strong><span>Public DeepGEMM SM100 forward path; training backward remains open.</span></div>
      </div>
      <div class="flow" aria-label="MegaMoE execution flow">
        <div class="flow-step"><b>Route</b><small>top-k outside the kernel</small></div>
        <div class="flow-step"><b>Dispatch</b><small>remote pull into L1 ring</small></div>
        <div class="flow-step"><b>Linear1</b><small>UMMA to TMEM</small></div>
        <div class="flow-step"><b>SwiGLU</b><small>BF16 activation and FP8 requant</small></div>
        <div class="flow-step"><b>Linear2</b><small>UMMA to TMEM</small></div>
        <div class="flow-step"><b>Combine</b><small>remote push, barrier, local reduce</small></div>
      </div>
      <div class="callout">Markdown remains the source of truth. This HTML is a
      generated snapshot for continuous reading and quick navigation.</div>

      <h3 id="source-docs">Source Documents<a class="anchor" href="#source-docs">#</a></h3>
      <h4>Guide Documents</h4>
      <div class="source-grid">{guide_cards}</div>
      <h4>Core Documents</h4>
      <div class="source-grid">{core_cards}</div>
      <h3 id="appendix-docs">External Hardware Documents<a class="anchor" href="#appendix-docs">#</a></h3>
      <div class="source-grid">{hardware_cards}</div>
    </section>

{body}
  </main>
</div>
</body>
</html>
"""


def main() -> None:
    missing = [doc.path for doc in [*GUIDE_DOCS, *CORE_DOCS] if not doc.path.exists()]
    if missing:
        raise SystemExit("Missing source files:\n" + "\n".join(str(path) for path in missing))

    renderer = MarkdownRenderer(collect_toc=False)
    sections = [synthesis_section(), visuals_section(), source_notes_intro()]
    for doc in [*GUIDE_DOCS, *CORE_DOCS]:
        sections.append(render_doc(renderer, doc))

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = html_page(toc_html(), "\n".join(sections), generated)
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
