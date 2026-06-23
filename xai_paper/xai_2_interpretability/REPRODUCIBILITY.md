# Reproducibility appendix — Paper 2 (ground-truth interpretability)

> **What this is.** Everything a reader needs to reproduce *every number* in the
> P2 (`xai_2_interpretability`) paper: the substrate + commit pins, the result-record
> schema (SPEC §R) and where the records live, the seeds + exact run command per
> phase, the packaged benchmark (so a third party can score a new method end-to-end),
> the ROM provenance (SHA-256 + AutoROM; ROMs are **not** redistributed), and the
> figure-regeneration scripts.
>
> **Read-only doc.** Reproduces what is committed; it does not change any experiment
> code or result. The committed §R records are the source of truth — the leaderboard
> (E6-1), the benchmark (E6-2), the faithful-method demo (E6-3) and all figures are
> **pure reads** over them.

---

## 1. Substrate + commit pins

The subject under study is the **Atari VCS itself** (chip + program + game logic),
run on two bit-exact differentiable ports validated against the `xitari` C++ reference.

| Component | Pin | Role |
|---|---|---|
| **Repo** | this checkout's `git rev-parse HEAD` | the frozen snapshot that produced the records |
| **jutari** (Julia/Zygote) | `jutari/` at repo HEAD (`JuTari` v0.0.1) | the experiment substrate — every Phase A/B/C runner is `julia --project=<repo>/jutari …` |
| **jaxtari** (JAX) | `jaxtari/` at repo HEAD (`jaxtari` v0.0.1, `requires-python >=3.10`, `jax>=0.4.30`) | cross-check port; benchmark + leaderboard + figures run on its venv (`numpy`/`json` only) |
| **xitari** | `xitari/` at repo HEAD | the bit-exact reference oracle the ports are validated against |
| **Julia** | 1.12.x (authoring env: 1.12.6); deps pinned by `jutari/Project.toml` | jutari runtime |

**Paper-1 conformance (the foundation this paper rests on).** jutari ≡ xitari
**bit-for-bit on all 64 ALE games — 64/64 RAM AND 64/64 screen** (README §"Conformance
vs xitari"; `tools/rom_sweep/results_jutari_ram.md`, `results_jutari_screen.md`;
in-repo gates **PXC1** xitari↔ports trace + **PXC2** jaxtari≡jutari cross-check;
seed patch `tools/xitari_conformance_seed.patch`). The key enabler for the oracle is
that the **SOFT-STE forward is bit-exact to the HARD path** at any finite temperature,
so exact interventions and gradients are GPU-batchable without leaving the conformance
window. Every §R record carries its own `commit` field (the records were produced
incrementally across the sprints; the pin above is the reproduction target, the
per-record `commit` documents provenance).

**Conformance gate (re-run before trusting any number):**
```bash
cd jutari && julia --project=. -e 'using Pkg; Pkg.test()'        # ~20 s, RAM/screen conformance
cd jaxtari && .venv/bin/python -m pytest -q                       # full suite (PXC1/PXC2)
```

---

## 2. The result-record schema (SPEC §R) + where the records live

Every experiment writes a **self-describing record** `out/<phase>/<exp>_<game>[_<regime>].json`
in the SPEC §R schema, with any heavy arrays in a sibling `.npz`:

```
{paper, phase, method, game, frame/state, target_output, metric_name, value,
 ci/stderr, n, seed, where, commit, oracle_ref, timestamp}   + extra{…}  + sibling .npz
```

The `extra{}` block carries the per-record specifics, including — for scorable records —
`cause_names` and `oracle_abs_delta_per_cause` (the **exact `|Δy(u)|` intervention map**),
plus the correctness-triad block `extra.triad.{F,S,M}` where the method reports S/M.
This is what makes the leaderboard (E6) a pure read and the benchmark oracle ROM-free.

**Where the records live** (counts at this HEAD):

| Path | Records | Content |
|---|---|---|
| `tools/xai_study/ground_truth/out/` | 3 | the §1 intervention/gradient/cross-check oracle records |
| `tools/xai_study/phaseA_kording/out/` | 54 | Phase A — Jonas&Kording battery (`A1…A8`) |
| `tools/xai_study/phaseB_attribution/out/` | 166 | Phase B — every attribution/XAI method × core games × regimes |
| `tools/xai_study/phaseC_mechanistic/out/` | 72 | Phase C — mechanistic interp (patching, SAE, probing, circuits) |
| `tools/xai_study/compare/out/` | leaderboard.{json,csv,md} + faithful_demo.{json,md,npz} | E6-1 leaderboard + E6-3 headline demo (pure reads) |
| `tools/xai_study/compare/benchmark/out/` | 14 | the packaged benchmark's `magnitude_proxy` demonstration records |
| `tools/xai_study/t3/out/` | candidates_*/discovered_*/verified_* | T3 game-concept labels (§2) |

The correctness triad (SPEC §0 / `experiment_design.md` §0) the records are scored on:
**F** Faithful (claimed causes are the *true* causes — vs the §1 oracle), **S** Sufficient
(predicts `y` under held-out interventions), **M** Minimal (parsimonious / right level).
F is always measured against the §1 oracle, never against another method.

---

## 3. Seeds + exact run commands per phase

**Seeds.** All committed records use `seed = 0` (deterministic replay; the differentiable
substrate is deterministic, so seeded methods — `random` baselines, RISE masks, SmoothGrad
noise, Expected-Gradients reference sampling — reproduce exactly). The `state` field encodes
the deterministic replay-to-state as `f<start>+<window>` (e.g. `f120+30` = boot to frame 120,
then a 30-frame window) or a named trajectory (`traj_60f_ram_noop`); both stay inside the
Paper-1 bit-exact conformance horizon (≈30-frame RAM / 60-frame screen).

**Generic invocation.** All experiment runners are Julia on the jutari substrate
(`<repo>` = this checkout's absolute root; jaxtari eager is ≈205× slower, so the
experiments run on jutari per SCRUM §5/§7):

```bash
julia --project=<repo>/jutari <runner>.jl [--games core] [--selftest]
```

`--games core` = the 6-game headline set (`pong breakout space_invaders seaquest
ms_pacman qbert`, fixed in `tools/xai_study/common/game_set.{md,json}`, SPEC §G);
`--selftest` runs the self-check only (writes nothing).

### Oracle (E1 — the ground truth everything is scored against)
```bash
julia --project=<repo>/jutari tools/xai_study/ground_truth/oracle_intervene.jl --game <g>   # exact |Δy(u)|
julia --project=<repo>/jutari tools/xai_study/ground_truth/oracle_grad.jl                    # ∂y/∂u + IG, content path
julia --project=<repo>/jutari tools/xai_study/ground_truth/oracle_xcheck.jl                  # intervention↔gradient corr
```

### T3 labels (E2 — game-concept labels, verified by intervention)
```bash
python3 tools/xai_study/t3/import_labels.py                                                  # candidate RAM→concept maps
julia --project=<repo>/jutari tools/xai_study/t3/verify_labels.jl                            # upgrade to verified-causal
julia --project=<repo>/jutari tools/xai_study/t3/discover_labels.jl                          # breadth-set discovery
```

### Phase A — Jonas & Kording battery (E3)
```bash
julia --project=<repo>/jutari tools/xai_study/phaseA_kording/pilot_si.jl --selftest          # self-check
julia --project=<repo>/jutari tools/xai_study/phaseA_kording/A1_connectomics.jl
julia --project=<repo>/jutari tools/xai_study/phaseA_kording/A2_lesions.jl --games core
julia --project=<repo>/jutari tools/xai_study/phaseA_kording/A3_tuning.jl
# …A4_correlations.jl · A5_lfp.jl · A6_granger.jl · A7_dimred.jl · A8_wholestate.jl (same form)
```

### Phase B — attribution / XAI (E4) — one runner per method
```bash
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl       # pilot (IG vs oracle)
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/saliency.jl       --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/gradxinput.jl     --games core   # --output content|position
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/guided_backprop.jl --games core --sanity
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/smoothgrad.jl     --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/expected_gradients.jl
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/occlusion.jl      --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/perturbation.jl   --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/rise.jl           --games core   # N=500 masks, ~14.5 min
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/lime.jl           --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/kernelshap.jl     --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/counterfactual.jl --games core
julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/na_audit.jl                  # N/A-method writeup (--md)
```

### Phase C — mechanistic interpretability (E5)
```bash
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/pilot_patch_sae.jl
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/activation_patching.jl --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/attribution_patching.jl --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/path_patching.jl   --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/acdc.jl            --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/das.jl             --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/linear_probing.jl  --games core   # + --selftest
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/logit_lens.jl      --games core
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/sae.jl
julia --project=<repo>/jutari tools/xai_study/phaseC_mechanistic/dictionaries.jl
```

### E6 — cross-tradition leaderboard + faithful-method demo (pure reads, no ROM)
```bash
python3 tools/xai_study/compare/leaderboard.py        # → compare/out/leaderboard.{json,csv,md}
python3 tools/xai_study/compare/faithful_demo.py      # → compare/out/faithful_demo.{json,md,npz}
```
The leaderboard re-orients each committed record's `value`/`extra.triad.F` onto the two
reporting axes (faithfulness X vs a transparent plausibility proxy Y) and runs an
embedded self-check; it does **not** re-run any experiment.

---

## 4. The packaged benchmark (E6-2) — score a new method end-to-end

`tools/xai_study/compare/benchmark/` packages the four pieces a third party needs to
score *one* interpretability method against the §1 ground-truth oracle and get a
faithfulness number directly comparable to the leaderboard:

| Piece | File | What it is |
|---|---|---|
| TASK set | `tasks.py` | 6 core games × {content, position, ball_pixel} = **14 scorable tasks** (4 degenerate (game,regime) pairs excluded — constant oracle column) |
| ORACLE | `oracle.py` | the §1 ground-truth causal map `{cause → |Δy(u)|}`, read from the committed §R records (**no ROM needed**) |
| METRICS | `metrics.py` | F = `max(0, pearson_corr(attr, |Δy_oracle|))` clipped to [0,1] + spearman + precision@k + (optional, ROM-needing) deletion/insertion AUC + the F/S/M triad |
| EXAMPLE | `example_method.py` + `run.py` | the plug-in contract + bundled methods (`oracle_copy`, `uniform`, `random`, `magnitude_proxy`) |

**Quick start** (any Python ≥3.9, numpy/json only; canonical interpreter is
`jaxtari/.venv/bin/python`, but a bare `python3` works since there are no extra deps):
```bash
PY=jaxtari/.venv/bin/python   # or: PY=python3
$PY -m tools.xai_study.compare.benchmark.run --self-test                       # validate the benchmark
$PY -m tools.xai_study.compare.benchmark.run --list-tasks                      # the 14-task set
$PY -m tools.xai_study.compare.benchmark.run --method oracle_copy --game pong  # positive control (F=1.0)
$PY -m tools.xai_study.compare.benchmark.run --method magnitude_proxy          # the bundled demonstration → out/
$PY -m tools.xai_study.compare.benchmark.run --method my_pkg.my_mod:my_method  # plug in YOUR method
```

A method is any callable `method(task, oracle) -> {"attribution": {cause: float}, "S": float|None, "M": float|None}`.
Records land in `benchmark/out/<method>_<game>_<regime>.json` in the §R schema (with
`extra.triad.F`), so the leaderboard reads them like any phase record. The self-test
asserts: `oracle_copy` is F==1 on all 14 tasks; `uniform` is at the floor (corr 0);
`magnitude_proxy` gives a finite F on pong/content (≈0.271); the oracle scores corr==1
against itself; ≥6 tasks exposed. **Verified passing at this HEAD** (`SELF-CHECK: PASS`).

The full machine-readable contract is `benchmark/manifest.json`; the human guide is
`benchmark/README.md`.

---

## 5. ROM provenance (not redistributed)

Atari 2600 ROMs are **gitignored and never committed** (SCRUM §7). Two facts make this
clean:

1. **Scoring needs no ROM.** The oracle ground truth is read from the committed §R
   records (`extra.cause_names` + `extra.oracle_abs_delta_per_cause`), produced by the
   exact intervention oracle on the real ROM. The whole benchmark + leaderboard +
   figures run offline.
2. **For the optional live re-run** (deletion/insertion AUC, which re-runs the real ROM)
   each core ROM is referenced by **SHA-256 + size + AutoROM name only**, never by bytes,
   in `tools/xai_study/compare/benchmark/rom_manifest.json`:

   | game | sha256 | bytes | AutoROM name |
   |---|---|---|---|
   | pong | `41623e3c…ec96d3` | 2048 | `pong` |
   | breakout | `376323f0…0c6fd5` | 2048 | `breakout` |
   | space_invaders | `7224b174…ced0301` | 4096 | `space_invaders` |
   | seaquest | `fbc29f46…d2ee43` | 4096 | `seaquest` |
   | ms_pacman | `dde0b43c…280b14` | 8192 | `ms_pacman` |
   | qbert | `3257221…194b76` | 4096 | `qbert` |

   (full 64-char hashes in `rom_manifest.json`).

**Obtain ROMs** (then verify against the manifest hashes):
```bash
pip install autorom && AutoROM --accept-license          # the licensed ALE/Gymnasium ROM set
# or place them in place (gitignored): xitari/games/Atari-2600-VCS-ROM-Collection/ROMS/
# tools/xai_study/common/loader.resolve_rom(name) resolves + you verify SHA-256
```
This mirrors Paper 1 and `document_check.md` §D (Data availability).

---

## 6. Figure regeneration

Each figure is a standalone script under `paper/figures/` (one `.py` → one `.pdf`,
disjoint). They are **pure reads** over the committed records / leaderboard (`matplotlib`
+ `numpy`); none re-runs an experiment. Run from the repo root:

```bash
PY=jaxtari/.venv/bin/python   # needs matplotlib + numpy
$PY xai_paper/xai_2_interpretability/paper/figures/fig1_platform_oracle.py
$PY xai_paper/xai_2_interpretability/paper/figures/fig2_faithfulness_vs_plausibility.py
$PY xai_paper/xai_2_interpretability/paper/figures/fig3_phaseA_battery.py
$PY xai_paper/xai_2_interpretability/paper/figures/fig4_attribution_vs_mechanistic.py
$PY xai_paper/xai_2_interpretability/paper/figures/fig5_representativeness_map.py
$PY xai_paper/xai_2_interpretability/paper/figures/fig6_failure_taxonomy.py
```
Each writes its `.pdf` next to the `.py` and prints `[OK] wrote …`.

| Figure | Reads | Shows |
|---|---|---|
| fig1 platform & oracle | `ground_truth/out/oracle_*` | platform + the §1 oracle |
| fig2 faithfulness vs plausibility | `compare/out/leaderboard.json`, `faithful_demo.json` | the headline cross-tradition axes |
| fig3 Phase-A battery | `phaseA_kording/out/A{1..8}_*`, leaderboard | the Kording battery scored |
| fig4 attribution vs mechanistic | `phase{B,C}/out/*`, leaderboard, faithful_demo | Phase B vs Phase C |
| fig5 representativeness map | `phase{B,C}/out/*`, leaderboard, faithful_demo | VCS↔NN failure-mode map |
| fig6 failure taxonomy | leaderboard, faithful_demo | failure taxonomy |

The paper itself builds from `paper/main.tex` (`pdflatex` + `bibtex`), pulling the PDFs
in `paper/figures/`.

---

## 7. End-to-end reproduction (one path)

1. Check out this repo at the HEAD pin; run the conformance gate (§1).
2. (Optional) regenerate the oracle on the real ROM (needs ROMs via §5):
   `julia --project=<repo>/jutari tools/xai_study/ground_truth/oracle_intervene.jl --game pong`.
3. Re-run any phase runner (§3) — it writes §R records into that phase's `out/`.
4. `python3 tools/xai_study/compare/leaderboard.py` — re-derives the leaderboard from
   the records (pure read; embedded self-check must PASS).
5. Score a new method against the same oracle: `… benchmark.run --method <yours>` (§4).
6. Regenerate the figures (§6).

Every number in the paper traces to a committed §R record; steps 3–6 regenerate the
derived artifacts from those records without touching the emulator core.
