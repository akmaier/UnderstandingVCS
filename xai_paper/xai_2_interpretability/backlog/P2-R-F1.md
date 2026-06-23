---
id: P2-R-F1
title: Redraw Fig 1 (platform/oracle) — jutari(Julia), plausibility-proxy, split/declutter, fonts
epic: RV (Revision — figure)
status: done
sprint: 10
owner:
where: local
depends_on: []
file_scope:
  - paper/figures/fig1_platform_oracle.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 1 per figure_detail_pass "Fig. 1" + improvement_instructions P4#38. Edit ONLY the .py
(regenerate fig1_platform_oracle.pdf). Apply:
- **jutari label (BLOCKER):** line 155 "jutari (JAX) · jaxtari (JAX)" → "jutari (Julia) · jaxtari
  (JAX)" (or "jutari (Julia/Zygote) · jaxtari (JAX/XLA)").
- **plausibility proxy (BLOCKER):** lines 28–29, 286 "human-plausibility"/"plausibility" → label
  the Y axis / reporting axis "plausibility proxy"; remove any "human" framing.
- **Declutter / split:** split into three clean columns (Substrate / Oracle / Scoring) or two
  panels (1a substrate+oracle, 1b scoring triad); remove most in-box prose; short labels only
  ("Exact re-run", "do(u:=v')", "Δy", "F: causes", "S: behavior", "M: parsimony"); make
  "oracle" vs "method under test" visually distinct.
- **Discrete-index caveat** as a separate small callout under the oracle, not nested in the
  gradient panel.
- **Remove source-path microtext:** line 323 (oracle: tools/xai_study/...) and any other
  provenance string → drop from figure body (provenance goes to SUPP).
- **Fonts:** minimum 8 pt for all in-figure text; legend outside or replaced by panel headers.
- **Global style guide:** colorblind-safe palette, no text boxes over data, legends outside axes,
  embedded fonts in the exported PDF.

## Definition of Done
- [x] regenerates: `python paper/figures/fig1_platform_oracle.py` → updated fig1_platform_oracle.pdf (self-check OK, 192 KB, %PDF-)
- [x] grep clean in the .py: no "jutari (JAX)", no "human plausibility" in drawn text, no source paths in the drawn figure (now "§3.2 oracle"; provenance microtext removed)
- [x] all in-figure text ≥ 8 pt (min drawn font 8.0 pt; panel headers 10.4 pt); oracle (vermillion, solid, heavy) vs method-under-test (grey, dashed) visually distinct; PDF has embedded fonts (pdffonts: emb=yes/sub=yes for all 6 faces)
- [x] nothing outside `file_scope` changed (only fig1_platform_oracle.py + .pdf + this item)
- [x] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [x] `status: done`

## Notes / handoff
CAPTION RULE: figure items edit ONLY the .py; the Fig.1 caption lives in sections/03_methods.tex
and is owned by P2-R-S03-methods. HANDOFF to S03: shorten the Fig.1 caption, say "plausibility
proxy" and "jutari (Julia)", move precision/recall/sign/magnitude detail to caption/SUPP.

### Done — what changed in the .py (R2)
Redrawn as three clean columns (1 SUBSTRATE / 2 ORACLE (§3.2) / 3 SCORE), one idea per column.
- jutari label fixed: "jutari (Julia) · jaxtari (JAX)".
- plausibility wording: in-figure axis callout now "axes: faithfulness vs plausibility proxy";
  "human" framing removed from drawn text.
- short labels only: "Exact re-run", "do(u:=v')", "Δy", "F: causes", "S: behavior",
  "M: parsimony"; prose paragraphs removed from boxes.
- bottom legend strip DELETED; colour identity now carried by direct coloured panel-header tabs
  (substrate=blue, oracle=vermillion, score=brown) → no legend competing with the caption.
- oracle vs method-under-test made visually distinct: oracle box = vermillion, solid, heavy
  (lw 2.2); "any XAI method → Ê" entry box = grey, dashed, labelled "method under test".
- discrete-index caveat is now ONE separate dashed callout under the oracle column (no longer
  nested in the gradient panel).
- ALL source-path / data-provenance microtext removed from the figure body (was the
  tools/xai_study/... oracle_intervene.{jl,py} footnote) → goes to caption/SUPP.
- all in-figure text ≥ 8 pt; Okabe-Ito palette (caveat purple darkened to #9b3d7a for contrast).

### fig_pass "Fig. 1" issues resolved
Blockers: jutari (JAX)→(Julia); "human plausibility"→"plausibility proxy". Major: split into
3 clean columns; removed in-box prose; legend removed (panel-header tabs instead); clean
single-label inter-panel lanes ("outputs y / + causes u", "ground truth"); F/S/M short labels;
gradient-panel declutter; discrete-index caveat separated; provenance microtext removed. Also
fixed two overlaps introduced during layout (column top boxes vs panel tabs) and one text-clip
(danger-zone line) — verified gone via pdftoppm @300dpi.

### CAPTION CHANGE the owning section (P2-R-S03-methods, 03_methods.tex) must make
The figure no longer prints: the legend, the precision/recall/sign/magnitude detail, the
"cross-check (1)↔(2)" annotation, or the provenance/source path. So the Fig.1 caption MUST now:
1. say "jutari (Julia) · jaxtari (JAX)" and "plausibility proxy" (matching the figure);
2. carry the F-detail (precision / recall / sign / magnitude) that was removed from the figure;
3. carry the artifact/provenance pointer ("Data/oracle: see Supplement; oracle defined in §3.2")
   that was removed from the figure body;
4. refer to the oracle as the "§3.2 oracle" / "intervention oracle" (NOT "Section 1 oracle").
