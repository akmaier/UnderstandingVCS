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
_(to be filled at the Sprint-1 barrier)_
