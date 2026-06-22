# P2 interpretability benchmark — tasks + oracle + metrics (P2-E6-2)

A **reusable benchmark** that lets a third party score *one* interpretability
method end-to-end against the P2 ground-truth causal **oracle** and obtain a
**faithfulness** score directly comparable to the cross-tradition leaderboard
(E6-1). It packages the four pieces a benchmark needs:

| Piece | File | What it is |
|---|---|---|
| **TASK set** | `tasks.py` | the 6 core games × output regimes (content / position / ball_pixel) with their candidate causes |
| **ORACLE** | `oracle.py` | the §1 ground-truth causal map `{cause → \|Δy(u)\|}`, read from the committed records (no ROM) |
| **METRICS** | `metrics.py` | faithfulness corr + spearman + precision@k + deletion/insertion AUC + the F/S/M triad |
| **EXAMPLE** | `example_method.py` + `run.py` | a runnable plug-in contract + bundled dummy methods scored end-to-end |

The benchmark is the operational form of `experiment_design.md` §0 (correctness
triad), §1 (oracle), §5 (Phase-B metrics). Faithfulness is always scored against
the §1 **oracle**, never against another interpretability method.

---

## Quick start

```bash
# from the repo root, using any Python ≥3.9 (numpy/json only; no extra deps)
PY=jaxtari/.venv/bin/python   # or your own interpreter

# 1. validate the benchmark itself (positive control + baselines)
$PY -m tools.xai_study.compare.benchmark.run --self-test

# 2. list the task set
$PY -m tools.xai_study.compare.benchmark.run --list-tasks

# 3. score a bundled example on one game
$PY -m tools.xai_study.compare.benchmark.run --method oracle_copy --game pong

# 4. score the demonstration (oracle-free) method over every task → writes records
$PY -m tools.xai_study.compare.benchmark.run --method magnitude_proxy

# 5. plug in YOUR method (a callable method(task, oracle) -> {attribution,S,M})
$PY -m tools.xai_study.compare.benchmark.run --method my_pkg.my_mod:my_method
```

Records land in `out/<method>_<game>_<regime>.json` in the SPEC §R schema (with
`extra.triad.F`), so the E6-1 leaderboard reads them like any phase record.

---

## The method contract (how to plug in)

A method is any callable

```python
def my_method(task, oracle) -> dict:
    # oracle.cause_names : the candidate causes (RAM cells, TIA regs, joystick)
    # oracle.abs_delta   : the TRUE |Δy(u)| per cause (DO NOT read this in a real
    #                      method — it is the ground truth you are scored against;
    #                      the oracle_copy example reads it only as a positive control)
    return {
        "attribution": {cause_name: float, ...},  # score EVERY cause (higher = more important)
        "S": float | None,   # optional Sufficiency leg (§0): predicts y under a held-out intervention
        "M": float | None,   # optional Minimality leg (§0): parsimony / right-level
    }
```

- **Faithfulness (F)** is always measured by the benchmark = `corr(attribution,
  |Δy_oracle|)` clipped to `[0,1]`.
- **S** and **M** are *method-reported* (a generic harness cannot synthesise
  them) and recorded when present; otherwise the record carries F only — exactly
  like the committed records (49 carry a full `extra.triad`, the rest carry F).

Programmatic use:

```python
from tools.xai_study.compare.benchmark import oracle, metrics, tasks
om = oracle.load_oracle("pong", "content")
out = my_method(tasks.get_task("pong", "content"), om)
score = metrics.score_method(out, om, k=3)   # -> {pearson_corr, precision_at_k, faithfulness, triad, ...}
```

---

## The task set

6 core games (`game_set.json` §G) × 3 output **regimes**:

- **content** — a content/colour output; `∂y/∂u` is defined, so gradient methods
  *can* succeed.
- **position** — a discrete sprite-position/index output whose **naive gradient
  is zero** (`experiment_design.md` §1); only intervention/causal methods recover
  it. This regime is the headline contrast.
- **ball_pixel** — a content pixel on the ball band (the Pong running example).

`run.py --list-tasks` prints the live set. **14 scorable tasks** ship; 4
(game, regime) pairs are **excluded as degenerate** — their oracle column is
constant/all-zero within the bit-exact conformance horizon (no probed cause
affected that pixel), so correlation carries no signal and even a perfect oracle
copy would score 0. They are listed (not hidden) in `manifest.json`.

---

## The oracle interface

`oracle.load_oracle(game, regime)` returns an `OracleMap` with:

- `cause_names`  — the ordered candidate causes,
- `abs_delta`    — the **exact** `|Δy(u)|` per cause (the §1 intervention oracle),
- `top_k(k)`     — the true causal top-k,
- `target_output`, `oracle_ref`, `commit`, `scorable`.

The ground truth is read straight from the committed Phase-B §R records
(`tools/xai_study/phaseB_attribution/out/<method>_<game>_<regime>.json`,
fields `extra.cause_names` + `extra.oracle_abs_delta_per_cause`) — these were
produced by the exact intervention oracle on the real ROM. **No ROM is needed to
score a method.**

### Regenerating the oracle

To add a new game/output, run the Julia generator on the real ROM:

```bash
julia --project=<repo>/jutari tools/xai_study/ground_truth/oracle_intervene.jl --game <g>
```

(SOFT-STE forward is bit-exact to this HARD map; Paper-1 64/64 conformance.)

---

## The metrics

`metrics.score_method(method_output, oracle, k=3, rerun_fn=None)` returns the
§R metric block. The formulas mirror the committed Julia scorer
(`tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl`):

- **Offline** (no ROM): `pearson_corr`, `spearman_corr`, `precision_at_k`,
  `faithfulness` (= `max(0, pearson_corr)` clipped to `[0,1]`).
- **Live** (optional, needs the ROM): `deletion_auc`, `insertion_auc` via a
  `rerun_fn(order, mode)` hook that re-runs the real emulator (deletion = occlude
  most-attributed first → small AUC if faithful; insertion = add back → large AUC
  if faithful). Off by default; the benchmark is fully runnable offline.
- **Triad** (§0): `F` (benchmark-measured), `S`/`M` (method-reported),
  `FSM_composite` (geometric mean when all three present).

---

## ROM non-redistribution

Atari 2600 ROMs are **not** shipped in this repo (gitignored, used in place —
SCRUM §7). **Scoring needs no ROM** (the oracle ground truth is committed). The
optional live deletion/insertion AUC references each core ROM by **SHA-256 +
AutoROM name** in `rom_manifest.json` — never ROM bytes. Obtain ROMs via
`AutoROM --accept-license` or place them under
`xitari/games/Atari-2600-VCS-ROM-Collection/ROMS/`; `loader.resolve_rom(name)`
resolves them and you can verify against the manifest hashes.

---

## Files

```
benchmark/
├── README.md          ← this file
├── manifest.json      ← machine-readable task list + oracle/metric API + ROM policy
├── rom_manifest.json  ← per-ROM SHA-256 + AutoROM name (no ROM bytes)
├── __init__.py
├── tasks.py           ← TASK set (6 core games × regimes)
├── oracle.py          ← ORACLE interface (the §1 ground-truth causal map)
├── metrics.py         ← METRIC definitions (corr / del-ins AUC / p@k / F-S-M)
├── example_method.py  ← bundled plug-in examples (oracle_copy / uniform / random / magnitude_proxy)
├── run.py             ← runnable entry point (python -m ...run)
└── out/               ← scored records (the magnitude_proxy demonstration, 14 records)
```

## Self-check

`python -m tools.xai_study.compare.benchmark.run --self-test` asserts:
the `oracle_copy` positive control is **F == 1** on every task; the `uniform`
baseline is at the **floor (corr 0)**; the oracle-free `magnitude_proxy` produces
a **finite faithfulness** on pong/content (≈ 0.27); the oracle interface scores
**corr == 1** against itself; and the benchmark exposes **≥ 6 tasks**. All pass.
