# Sprint 1 — Foundation + intervention oracle

## Goal
Build and **verify** the experiment foundation that everything else depends on: the
harness, the state recorder, T3 import, the cluster templates, and the **exact
intervention oracle** (E1-1) — the linchpin every method is scored against.

## Planning
Dependency reality: Sprint 1 is mostly a **chain** (E0-1 → E0-2 → E1-1), with two
independent items that parallelise. Scoped to the foundation; the oracle-*dependent*
items (E1-2 gradient oracle, E1-3 cross-check, E0-4 audit, pilots E3-0/E4-0/E5-0) are
**deferred to Sprint 2** — we verify E1-1 at this Review before building on it.

Items (disjoint file-scopes; run as one Workflow, ≤3 parallel):
- **Wave A (parallel, no oracle dep):** `P2-E0-1` harness skeleton (`common/`),
  `P2-E2-1` T3 candidate-label import (`t3/`), `P2-E0-3` cluster sbatch templates
  (`tools/cluster/`). Disjoint trees.
- **Wave B:** `P2-E0-2` state/trajectory recorder (`common/record_state.*`) — deps E0-1.
- **Wave C:** `P2-E1-1` intervention oracle (`ground_truth/oracle_intervene.*`) — deps E0-1, E0-2.

Env facts given to agents: worktrees lack gitignored ROMs/venvs → use the **primary
absolute paths** (ROMs at `xitari/games/Atari-2600-VCS-ROM-Collection/ROMS/`, jaxtari
`.venv`, `julia --project=.../jutari`); never modify the emulator core.

## Review
**Foundation: DONE (4/4), verified.** E0-1 harness, E0-2 recorder, E0-3 cluster templates,
E2-1 T3 candidate labels — built, pushed, present; the shared jaxtari venv is intact
(numpy 2.4.6 / jax 0.10.1 / smoke OK) after E2-1's transient ocatari→numpy churn (restored).

**E1-1 intervention oracle: code written (`fc9316c`) but BLOCKED — NOT verified.**
`oracle_intervene.py` + `test_oracle.py` are implemented and logic-checked (reads Pong
$0D/$0E, 18 causes set/occlude/replace, bit-exact assertion + §R writer), but it could not
RUN: a single **jaxtari eager** Pong boot is ~10 min (E0-1's test = 1305 s for 2 boots; an
SM diagnostic timed out at 280 s for one boot in the *primary*). The oracle needs ~3 boots
+ interventions → it stalled even at 28 min. The bit-exact assertion never executed; no
causal-map artifact exists. Status: **blocked**.

**Root cause + decision (PAUSED for PO).** The foundation was built on the **jaxtari eager**
HARD path, which is ~**205× slower than jutari** — ~10 min/boot makes it unusable for the
experiment volume (B/C = methods × games × many interventions). jaxtari's speed lives only
in its **jit+vmap SOFT-STE GPU-batch** path, not eager HARD stepping. **Recommendation:**
run experiments on **jutari (Julia)** — the proven fast real-ROM path (`tools/xai_si_gradient`
already does real-Pong gradients there) with native Zygote differentiability for the
gradient oracle — and reserve jaxtari for **GPU-batched SOFT sweeps on the cluster**. This
re-targets E0-1 (loader/replay), E0-2 (recorder), and E1-1 (oracle) onto jutari. **Awaiting
PO go-ahead before Sprint 2.**

**Retro / process fixes (apply on pivot):** (1) agents must NOT `pip install` into the
shared venv — E2-1 broke jax transiently (add SCRUM §7 gotcha; done in spirit). (2) make
"experiments run on **jutari** locally; jaxtari = **cluster GPU-batch** only" an explicit
ENV fact in SCRUM/SPEC. (3) heavy-RUN items need real ROMs/venv — worktrees lack gitignored
files; run them in the primary or provision the worktree.
