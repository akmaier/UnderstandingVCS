"""oracle_intervene — the EXACT intervention oracle (P2-E1-1).

The *primary* ground-truth instrument for Paper 2 (`experiment_design.md` §1,
`SPEC.md#E1`). For a VCS output ``y`` (a pixel, the score, a game event, a future
state) and a candidate cause ``u`` (a RAM cell, a TIA register, a joystick input,
...), it measures the **true causal effect**

    Δy(u) = y( do(u := v') )  −  y( baseline )

by *intervening* on ``u`` at a target frame, re-running the deterministic emulator
to the end of a short horizon, and reading the resulting output. The emulator is
bit-exact under a fixed action trace, so the un-intervened re-run reproduces the
baseline **byte-for-byte** — this module asserts that (RAM *and* screen identical
across two fresh envs) before trusting any Δ. That bit-exact re-run is exactly the
property the SOFT-STE forward path preserves (Paper-1 Corollary): SOFT at any
finite temperature is byte-identical to HARD, so the same {Δy(u)} map is what a
batched GPU SOFT-STE sweep would produce. This local entry point computes a small,
correct map; the exhaustive 128-byte × all-frames sweep is deferred to the cluster
(E0-3 templates) — see ``main()``'s note.

No world model is assumed: every Δ is a real re-run of the real ROM.

Mechanism (per cause ``u``):
  1. Fresh env, deterministic replay to ``target_frame``.
  2. Apply the intervention on ``u`` (set / occlude / replace / resample, or, for a
     joystick cause, swap the action at the target frame).
  3. Continue ``horizon`` frames with the SAME action tail.
  4. Read every registered output ``y`` and record Δy = y_intervened − y_baseline.

Outputs (``y``) for Pong (the running example, `experiment_design.md` §1):
  * ``p0_score``  = RAM[$0D] (index 13) — the agent/"cpu" score.
  * ``p1_score``  = RAM[$0E] (index 14) — the opponent/"human" score.
  * ``ball_pixel``= the palette index at a fixed framebuffer cell over the ball
                    band (a *content* pixel output, per §1).

Run:  see ``main()`` / the module docstring at the bottom and the README.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Package import (PYTHONPATH=<worktree>); falls back to a path shim if run as a
# bare script from elsewhere.
try:
    from tools.xai_study.common import loader, replay, results, seeds
except ModuleNotFoundError:  # pragma: no cover - convenience for direct runs
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.xai_study.common import loader, replay, results, seeds


# --- Pong constants (xitari Pong.cpp; see jaxtari/games/pong.py) -------------
PONG_P0_SCORE_IDX = 0x0D   # RAM index 13 — agent / "cpu" score
PONG_P1_SCORE_IDX = 0x0E   # RAM index 14 — opponent / "human" score
COLUP1_REG = 0x08          # TIA COLUP1 — Pong's ball/right-paddle colour register

OUT_DIR = Path(__file__).resolve().parent / "out"

# Default candidate-cause file (from E2-1 import). Resolved across worktree +
# primary checkout because t3/out/ is produced locally.
_CANDIDATES_REL = "tools/xai_study/t3/out/candidates_pong.json"


# ============================================================================
# Outputs y(state)
# ============================================================================
@dataclass(frozen=True)
class Output:
    """A scalar VCS output ``y`` read from a replayed Snapshot."""
    name: str
    kind: str                      # "score" | "pixel"
    read: Callable[[replay.Snapshot], float]
    note: str = ""


def _ball_pixel_cell(baseline: replay.Snapshot) -> Tuple[int, int]:
    """Pick a fixed (row, col) over Pong's ball band for the ball-pixel output.

    We choose the cell from the *baseline* framebuffer: the brightest non-border
    column on the most-populated playfield row inside the central band. Fixing it
    from the baseline keeps the output a stable function of state across all
    interventions (the oracle reads the SAME cell every time)."""
    scr = baseline.screen
    h, w = scr.shape
    # Central vertical band (skip the score header rows at the very top).
    r0, r1 = int(h * 0.25), int(h * 0.85)
    band = scr[r0:r1]
    # Per-row count of lit, non-wall pixels (walls are the bright top/edges).
    lit = (band != 0)
    row_counts = lit.sum(axis=1)
    # Most-active row in the band, then the median lit column on that row.
    row = int(np.argmax(row_counts)) + r0
    cols = np.flatnonzero(scr[row] != 0)
    if cols.size == 0:
        return row, w // 2
    col = int(cols[len(cols) // 2])
    return row, col


def pong_outputs(baseline: replay.Snapshot) -> List[Output]:
    """The three Pong outputs (two scores + one ball pixel)."""
    pr, pc = _ball_pixel_cell(baseline)
    return [
        Output("p0_score", "score",
               lambda s: float(int(s.ram[PONG_P0_SCORE_IDX])),
               "RAM[$0D] agent/cpu score"),
        Output("p1_score", "score",
               lambda s: float(int(s.ram[PONG_P1_SCORE_IDX])),
               "RAM[$0E] opponent/human score"),
        Output(f"ball_pixel@r{pr}c{pc}", "pixel",
               lambda s, r=pr, c=pc: float(int(s.screen[r, c])),
               f"palette index at framebuffer cell (row={pr}, col={pc})"),
    ]


# ============================================================================
# Causes u + interventions
# ============================================================================
@dataclass(frozen=True)
class Cause:
    """A candidate cause ``u`` and how to intervene on it.

    ``kind``:
      * ``ram``      — write byte ``value`` into RIOT RAM[index] at the target frame.
      * ``tia_reg``  — write byte ``value`` into TIA register file[index].
      * ``joystick`` — replace the action applied at the target frame with ``value``
                       (a do(action) intervention; the action-trace tail is kept).
    ``mode`` records the *semantic* of the chosen value for the record (set / occlude
    / replace / resample) so the causal map is self-describing.
    """
    name: str
    kind: str
    index: int                     # RAM idx / TIA reg / (unused for joystick)
    value: int                     # the do-value
    mode: str                      # "set" | "occlude" | "replace" | "resample"
    concept: str = ""              # T3 concept label (if any), for provenance
    note: str = ""


def _apply_ram(console, index: int, value: int):
    import jax.numpy as jnp
    new_ram = console.bus.ram.at[index].set(jnp.uint8(value & 0xFF))
    new_bus = console.bus._replace(ram=new_ram)
    return console._replace(bus=new_bus)


def _apply_tia_reg(console, index: int, value: int):
    import jax.numpy as jnp
    tia = console.bus.tia
    new_regs = tia.registers.at[index].set(jnp.uint8(value & 0xFF))
    new_tia = tia._replace(registers=new_regs)
    new_bus = console.bus._replace(tia=new_tia)
    return console._replace(bus=new_bus)


# ============================================================================
# Replay + intervene
# ============================================================================
def _fresh_env(game: str, *, rom_path: Optional[str] = None, seed: int = 0):
    """A freshly-reset, xitari-parity env (deterministic boot)."""
    return loader.load_game(game, rom_path=rom_path, seed=seeds.env_seed(seed))[0]


def _bare_env(game: str, *, rom_path: Optional[str] = None):
    """An env *without* the boot reset — used to host a restored console.

    The expensive part of a deterministic replay is the xitari-parity boot
    (60 NOOP + 4 RESET = 64 frames) plus the replay-to-target steps; doing that
    once and *restoring the captured console* into a bare env lets every
    intervention continue from the checkpoint without re-paying that cost. The
    console is an immutable NamedTuple (CPU + bus + RAM + TIA + RIOT + cart), so
    restoring it is exact — the continuation is byte-identical to a fresh replay.
    """
    return loader.load_game(game, rom_path=rom_path, reset=False)[0]


def _step(env, action: int) -> None:
    env.step(int(action))


def _snapshot(env, frame: int, actions: Sequence[int]) -> replay.Snapshot:
    return replay.Snapshot(
        frame=frame,
        ram=np.asarray(env.get_ram(), dtype=np.uint8),
        screen=np.asarray(env.get_screen(), dtype=np.uint8),
        actions=[int(a) for a in actions],
    )


def make_checkpoint(game: str, actions: Sequence[int], target_frame: int, *,
                    rom_path: Optional[str] = None, seed: int = 0):
    """Boot + deterministic-replay to ``target_frame`` once; return its console.

    The returned object is the exact emulator state at the intervention frame.
    Every cause/baseline continuation restores it into a bare env, so the boot
    and the to-target replay are paid exactly once for the whole sweep."""
    env = _fresh_env(game, rom_path=rom_path, seed=seed)
    for a in actions[:target_frame]:
        _step(env, a)
    return env.console


def _continue_from(console, game: str, tail: Sequence[int], *,
                   rom_path: Optional[str] = None) -> replay.Snapshot:
    """Restore ``console`` into a bare env, run ``tail`` frames, snapshot."""
    env = _bare_env(game, rom_path=rom_path)
    env._console = console
    for a in tail:
        _step(env, int(a))
    return _snapshot(env, len(tail), tail)


def run_baseline(game: str, actions: Sequence[int], target_frame: int,
                 horizon: int, *, rom_path: Optional[str] = None,
                 seed: int = 0, checkpoint=None) -> replay.Snapshot:
    """Un-intervened replay to ``target_frame + horizon``.

    If ``checkpoint`` (a console captured at ``target_frame``) is given, continue
    from it; otherwise boot + replay from scratch."""
    acts = list(actions)
    if checkpoint is None:
        checkpoint = make_checkpoint(game, acts, target_frame,
                                     rom_path=rom_path, seed=seed)
    return _continue_from(checkpoint, game,
                          acts[target_frame: target_frame + horizon],
                          rom_path=rom_path)


def assert_bit_exact(game: str, actions: Sequence[int], total: int, *,
                     rom_path: Optional[str] = None, seed: int = 0) -> None:
    """Two fresh un-intervened re-runs must be byte-identical (RAM and screen).

    This is the load-bearing correctness guarantee of the oracle: every Δy is a
    difference against a baseline that the emulator reproduces exactly, so a
    nonzero Δ is *caused* by the intervention and nothing else. Both re-runs boot
    + replay from scratch (no shared checkpoint) so the determinism of the *whole*
    pipeline — boot included — is what is asserted."""
    a = run_baseline(game, actions, total, 0, rom_path=rom_path, seed=seed)
    b = run_baseline(game, actions, total, 0, rom_path=rom_path, seed=seed)
    if not np.array_equal(a.ram, b.ram):
        diff = int((a.ram != b.ram).sum())
        raise AssertionError(
            f"bit-exact RAM re-run FAILED: {diff}/128 bytes differ across two "
            f"fresh '{game}' replays to frame {total}")
    if not np.array_equal(a.screen, b.screen):
        diff = int((a.screen != b.screen).sum())
        raise AssertionError(
            f"bit-exact SCREEN re-run FAILED: {diff} pixels differ across two "
            f"fresh '{game}' replays to frame {total}")


def run_intervention(game: str, actions: Sequence[int], target_frame: int,
                     horizon: int, cause: Cause, *,
                     rom_path: Optional[str] = None,
                     seed: int = 0, checkpoint=None) -> replay.Snapshot:
    """Replay to ``target_frame``, apply ``cause``, continue ``horizon`` frames.

    Restores the shared ``checkpoint`` (captured at ``target_frame``) when given,
    so only the ``horizon`` continuation is re-run per cause."""
    acts = list(actions)
    if checkpoint is None:
        checkpoint = make_checkpoint(game, acts, target_frame,
                                     rom_path=rom_path, seed=seed)
    if cause.kind == "joystick":
        # do(action): replace the action applied AT the target frame.
        tail = [cause.value] + list(acts[target_frame + 1: target_frame + horizon])
        return _continue_from(checkpoint, game, tail, rom_path=rom_path)
    # Intervene on the checkpoint console, then run the unchanged action tail.
    if cause.kind == "ram":
        console = _apply_ram(checkpoint, cause.index, cause.value)
    elif cause.kind == "tia_reg":
        console = _apply_tia_reg(checkpoint, cause.index, cause.value)
    else:
        raise ValueError(f"unknown cause kind: {cause.kind!r}")
    tail = acts[target_frame: target_frame + horizon]
    return _continue_from(console, game, tail, rom_path=rom_path)


# ============================================================================
# Causal map
# ============================================================================
def build_pong_causes(candidates_path: Optional[Path],
                      baseline: replay.Snapshot) -> List[Cause]:
    """A SMALL, bounded set of Pong causes (NOT all 128 bytes).

    Drawn from E2-1's candidate RAM bytes (ball/paddle/score cells) + a TIA
    colour register + a joystick input. For each RAM byte we use a SET/occlude
    intervention with a value far from the baseline so the effect is observable
    within the short horizon."""
    causes: List[Cause] = []

    # --- candidate RAM bytes (from E2-1 import; de-duplicated by ram_index) ---
    cand_idxs: List[Tuple[int, str]] = []
    if candidates_path and candidates_path.is_file():
        data = json.loads(candidates_path.read_text())
        seen = set()
        for c in data.get("candidates", []):
            idx = int(c["ram_index"])
            if idx in seen:
                continue
            seen.add(idx)
            cand_idxs.append((idx, str(c.get("concept", ""))))
    if not cand_idxs:
        # Fallback to the documented Pong cells if the import file is absent.
        cand_idxs = [(13, "enemy_score"), (14, "player_score"),
                     (49, "ball_x"), (54, "ball_y"),
                     (51, "player_y"), (50, "enemy_y")]

    for idx, concept in cand_idxs:
        base = int(baseline.ram[idx])
        # An occlude-to-zero plus a set-to-far value; for the score cells a
        # set-to-large value makes the score-pixel delta unambiguous.
        causes.append(Cause(
            name=f"ram[{idx}]:set", kind="ram", index=idx,
            value=(base + 17) & 0xFF, mode="set", concept=concept,
            note=f"RAM[{idx}] {concept} <- base+17"))
        causes.append(Cause(
            name=f"ram[{idx}]:occlude", kind="ram", index=idx,
            value=0, mode="occlude", concept=concept,
            note=f"RAM[{idx}] {concept} <- 0"))

    # --- a TIA register cause (ball/right-paddle colour) -> ball pixel --------
    causes.append(Cause(
        name="tia[COLUP1]:set", kind="tia_reg", index=COLUP1_REG,
        value=0x0E, mode="set", concept="ball_colour",
        note="TIA COLUP1 <- 0x0E (white) — content path to the ball pixel"))

    # --- a joystick input cause (do(action)) ---------------------------------
    # Action 2 = UP, 5 = DOWN in the ALE set; either moves the agent paddle.
    causes.append(Cause(
        name="joystick:UP", kind="joystick", index=-1, value=2,
        mode="replace", concept="agent_input",
        note="do(action := UP) at the target frame"))

    return causes


@dataclass
class CausalMap:
    """The computed {Δy(u)} map and its provenance."""
    game: str
    target_frame: int
    horizon: int
    seed: int
    output_names: List[str]
    output_notes: List[str]
    cause_names: List[str]
    cause_meta: List[Dict] = field(default_factory=list)
    y_baseline: Dict[str, float] = field(default_factory=dict)
    delta: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (causes, outputs)
    bit_exact: bool = False


def compute_causal_map(game: str, actions: Sequence[int], target_frame: int,
                       horizon: int, *, candidates_path: Optional[Path] = None,
                       rom_path: Optional[str] = None,
                       seed: int = 0, verbose: bool = False) -> CausalMap:
    """Compute the full {Δy(u)} causal map for Pong's outputs.

    Boots + replays to the target frame ONCE (``make_checkpoint``); the baseline
    and every intervention continue from that shared console, so the expensive
    xitari-parity boot is paid a single time for the whole sweep."""
    total = target_frame + horizon

    # 1) Bit-exact baseline guarantee (the load-bearing assertion). Runs the WHOLE
    #    pipeline twice from scratch (boot included) — no shared checkpoint here.
    assert_bit_exact(game, actions, total, rom_path=rom_path, seed=seed)

    # 2) One checkpoint at the intervention frame; reuse it everywhere below.
    checkpoint = make_checkpoint(game, actions, target_frame,
                                 rom_path=rom_path, seed=seed)
    # at_target = the state AT the intervention frame (horizon 0 from checkpoint).
    at_target = _continue_from(checkpoint, game, [], rom_path=rom_path)
    outputs = pong_outputs(at_target)

    # 3) Baseline outputs (continue the unchanged tail from the checkpoint).
    base_snap = run_baseline(game, actions, target_frame, horizon,
                             rom_path=rom_path, seed=seed, checkpoint=checkpoint)
    y_base = {o.name: o.read(base_snap) for o in outputs}

    # 4) Causes.
    causes = build_pong_causes(candidates_path, at_target)

    # 5) Δy(u) for every (cause, output) — each continues from the checkpoint.
    delta = np.zeros((len(causes), len(outputs)), dtype=np.float64)
    cause_meta: List[Dict] = []
    for i, cause in enumerate(causes):
        snap = run_intervention(game, actions, target_frame, horizon, cause,
                                rom_path=rom_path, seed=seed, checkpoint=checkpoint)
        for j, o in enumerate(outputs):
            delta[i, j] = o.read(snap) - y_base[o.name]
        if verbose:
            print(f"  [{i + 1}/{len(causes)}] {cause.name:<22} "
                  f"max|Δ|={float(np.max(np.abs(delta[i]))):.3f}", flush=True)
        cause_meta.append({
            "name": cause.name, "kind": cause.kind, "index": cause.index,
            "value": cause.value, "mode": cause.mode, "concept": cause.concept,
            "note": cause.note,
        })

    return CausalMap(
        game=game, target_frame=target_frame, horizon=horizon, seed=seed,
        output_names=[o.name for o in outputs],
        output_notes=[o.note for o in outputs],
        cause_names=[c.name for c in causes],
        cause_meta=cause_meta,
        y_baseline=y_base,
        delta=delta,
        bit_exact=True,
    )


# ============================================================================
# Persist (SPEC §R)
# ============================================================================
def write_causal_map(cmap: CausalMap, out_dir: Path = OUT_DIR) -> Path:
    """Write the causal map as a SPEC §R record (+ sibling .npz arrays).

    JSON: one record with the summary metric (max |Δy| on the score output) and
    the full per-(cause,output) map echoed under ``extra`` for human reading.
    NPZ:  ``delta`` (causes×outputs), plus index arrays for downstream tools.
    """
    out_dir = Path(out_dir)
    # Summary metric: the largest absolute score delta produced by any cause —
    # a single scalar that proves the oracle found a real causal effect.
    score_cols = [k for k, n in enumerate(cmap.output_names)
                  if cmap.output_names[k].endswith("score")]
    if score_cols:
        max_abs_score_delta = float(np.max(np.abs(cmap.delta[:, score_cols])))
    else:
        max_abs_score_delta = float(np.max(np.abs(cmap.delta)))

    rec = results.ResultRecord(
        phase="ground_truth",
        method="intervention_oracle",
        game=cmap.game,
        state=f"f{cmap.target_frame}+{cmap.horizon}",
        target_output="pong_score+ball_pixel",
        metric_name="max_abs_score_delta",
        value=max_abs_score_delta,
        n=len(cmap.cause_names),
        seed=cmap.seed,
        where="local",
        oracle_ref=f"oracle_intervene@{cmap.game}#score,ball_pixel",
        extra={
            "outputs": cmap.output_names,
            "output_notes": cmap.output_notes,
            "causes": cmap.cause_meta,
            "y_baseline": cmap.y_baseline,
            "bit_exact_rerun": cmap.bit_exact,
            "delta_map": {
                c: {o: float(cmap.delta[i, j])
                    for j, o in enumerate(cmap.output_names)}
                for i, c in enumerate(cmap.cause_names)
            },
            "scales_to_cluster_via": "tools/cluster/xai_*.sbatch (E0-3); batch "
                                     "the (cause × frame) grid via SOFT-STE — "
                                     "forward is bit-exact to this HARD map.",
        },
    )
    arrays = {
        "delta": cmap.delta,                                  # (causes, outputs)
        "cause_names": np.array(cmap.cause_names, dtype=object),
        "output_names": np.array(cmap.output_names, dtype=object),
        "y_baseline": np.array([cmap.y_baseline[o] for o in cmap.output_names],
                               dtype=np.float64),
    }
    return results.write_record(rec, out_dir, exp="oracle_pong_score",
                                arrays=arrays)


# ============================================================================
# CLI
# ============================================================================
def _resolve_candidates(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    # search worktree + primary checkout (t3/out is produced locally)
    for base in (loader.REPO, loader._primary_repo()):
        p = base / _CANDIDATES_REL
        if p.is_file():
            return p
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="pong")
    ap.add_argument("--target-frame", type=int, default=6,
                    help="frame to intervene at (small; inside the horizon)")
    ap.add_argument("--horizon", type=int, default=6,
                    help="frames to continue after the intervention")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", default=None,
                    help="path to candidates_<game>.json (default: auto-resolve)")
    ap.add_argument("--rom-path", default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    seeds.seed_everything(args.seed)
    cand = _resolve_candidates(args.candidates)
    actions = [0] * (args.target_frame + args.horizon)  # NOOP trace (deterministic)

    print(f"[oracle_intervene] game={args.game} target_frame={args.target_frame} "
          f"horizon={args.horizon} seed={args.seed}")
    print(f"[oracle_intervene] candidates: {cand}")
    cmap = compute_causal_map(
        args.game, actions, args.target_frame, args.horizon,
        candidates_path=cand, rom_path=args.rom_path, seed=args.seed,
        verbose=True)

    path = write_causal_map(cmap, Path(args.out_dir))
    print(f"[oracle_intervene] bit-exact re-run asserted: {cmap.bit_exact}")
    print(f"[oracle_intervene] baseline y: {cmap.y_baseline}")
    print("[oracle_intervene] sample Δy (cause -> {output: Δ}):")
    for i, c in enumerate(cmap.cause_names):
        row = {o: round(float(cmap.delta[i, j]), 3)
               for j, o in enumerate(cmap.output_names)}
        print(f"    {c:<22} {row}")
    print(f"[oracle_intervene] wrote {path}")
    print(f"[oracle_intervene] arrays  {path.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
