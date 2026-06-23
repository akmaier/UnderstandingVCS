# R2 → section-item caption handoffs

The R2 figure redraws removed in-figure prose, legends, per-metric keys, and citation names to
hit the 8 pt font floor and clear overlaps. The captions (owned by the section items, NOT the
figure items) must now carry that information. Each section item below MUST apply its handoff and
keep the figure↔paper-number map straight (figure FILES are numbered in creation order, NOT paper
order):

| Paper Fig | file | section file | section item |
|---|---|---|---|
| Fig 1 | fig1_platform_oracle | 03_methods.tex | P2-R-S03-methods |
| Fig 2 | fig3_phaseA_battery | 04_results_A.tex | P2-R-S04-resultsA |
| Fig 3 | fig4_attribution_vs_mechanistic | 05_results_B.tex | P2-R-S05-resultsB |
| Fig 4 (scatter, p18) | fig2_faithfulness_vs_plausibility | 07_results_compare.tex | P2-R-S07-compare |
| Fig 5 (taxonomy, p30) | fig6_failure_taxonomy | 07_results_compare.tex | P2-R-S07-compare |
| Fig 6 (VCS→NN, p31) | fig5_representativeness_map | 08_discussion.tex | P2-R-S08-discussion |

## P2-R-S03-methods — Fig 1 caption (`03_methods.tex`)
- Say "jutari (Julia) · jaxtari (JAX)" and "plausibility proxy"; oracle = "§3.2 intervention oracle".
- The figure no longer prints the legend, the F-detail, or the cross-check annotation — the caption
  must carry the F precision/recall/sign/magnitude detail and a provenance pointer
  ("Data/oracle: see Supplement; oracle defined in §3.2").

## P2-R-S04-resultsA — Fig 2 caption (`04_results_A.tex`)
- Shorten; takeaway first. "intervention oracle (§3.2)" (never §1/Section 1).
- Carry the per-metric meanings removed from the figure: A1 = data-flow-graph F1 vs oracle; A2 =
  lesion-importance ρ vs true role; A3 = 1 − spurious-tuning rate; A4 = coupling corr. vs true
  coupling; A5 = 1 − clock-explained var.; A6 = 1 − false-edge rate vs true data-flow; A7 = NMF
  matched-component fraction; A8 = minimality M.
- Whiskers = asymmetric 95% bootstrap CIs (leaderboard_ci.csv, R-UNC); all 8 Phase-A bars have one.
- Panel (b) is now a key-results TABLE (was callout cards) — reword any "callout"/"card" reference.

## P2-R-S05-resultsB — Fig 3 caption (`05_results_B.tex`)
- Drop the stale `\S1`; use "the intervention oracle (§3.2 / \S\ref{sec:methods})".
- "higher human-plausibility proxy" → "higher plausibility proxy" (never "human plausibility").
- Add an uncertainty sentence (Phase-C per-method 95% CIs; Phase-B family-level position-regime CI
  band). Note the figure shows the 22 Phase-B+C rows, a subset of the 30 methods + oracle = 31 rows.

## P2-R-S07-compare — Fig 4 (scatter) + Fig 5 (taxonomy) captions (`07_results_compare.tex`)
- **Fig 4 (fig2_faithfulness_vs_plausibility):** keep "plausibility proxy" + "intervention oracle
  (§3.2)"; state "30 methods + the oracle = 31 rows" verbatim; mention the new 95% bootstrap-over-
  games CI whiskers; shorten — the key findings now live in the in-figure table.
- **Fig 5 (fig6_failure_taxonomy):** keep `\includegraphics{fig6_failure_taxonomy.pdf}` (simplified
  main); shorten caption to <120–160 words; oracle = §3.2; "F ± 95% bootstrap CI"; cite the 7 works
  now in references.bib (R-REF); label NN parallels as documented analogues, not proof. The FULL
  version `fig6_failure_taxonomy_full.pdf` is committed — SUPP/integration adds it to the supplement.

## P2-R-S08-discussion — Fig 6 caption (`08_discussion.tex`)
- Shorten to one takeaway + a Supplement data pointer; calm wording (NN column = "documented
  analogue, not proven by this benchmark"); oracle = §3.2.
- In-figure citation names were removed — caption/text must cite the analogue works (Balduzzi 2017,
  Sundararajan 2017, Adebayo 2018, Anand 2019, Jain & Wallace 2019, Morcos 2018, Leavitt & Morcos
  2020, Hewitt & Liang 2019, Elhage 2022, Kindermans 2019, Sturmfels 2020) — all confirmed present
  in references.bib by R-REF.
- The FULL version `fig5_representativeness_map_full.pdf` is committed — SUPP/integration adds it to
  the supplement; the simplified `fig5_representativeness_map.pdf` stays in the main text (PO#4).

## Data reconciliation note for P2-R-S07-compare / P2-R-SUPP
`sparse_autoencoder` differs between records: leaderboard.json F=0.1947 vs leaderboard_ci.csv
F_mean=0.0605 / mean=0.1404. Figures plot leaderboard.json F with the csv ci_lo/ci_hi
([0.0, 0.314], which brackets 0.1947). S07/SUPP should reconcile the two aggregations (or state
which is canonical) so text, table, and figure agree.
