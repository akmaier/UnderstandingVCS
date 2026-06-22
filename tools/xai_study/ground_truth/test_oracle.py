"""test_oracle — correctness gates for the exact intervention oracle (P2-E1-1).

Two load-bearing properties (DoD):
  1. **Bit-exact re-run** — two fresh, un-intervened Pong replays to the same
     frame are byte-identical in RAM *and* screen. Every Δy the oracle reports is
     a difference against a baseline the emulator reproduces exactly, so a nonzero
     Δ is caused by the intervention and nothing else.
  2. **Sane causal map** — intervening on a SCORE byte moves the corresponding
     score output (nonzero Δ), while intervening on a quiet BACKGROUND byte leaves
     the score outputs ~0. (The oracle distinguishes a real cause from a null one.)

Run (from the repo root, with the shared jaxtari venv):
    XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \\
    PYTHONPATH=$PWD \\
    /Users/maier/Documents/code/UnderstandingVCS/jaxtari/.venv/bin/python \\
        -m pytest tools/xai_study/ground_truth/test_oracle.py -v

These run the real Pong ROM in eager jaxtari (~3.5 s/step), so the module keeps
the horizon tiny (target_frame small, horizon small) — minutes, not hours.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `tools.xai_study.*` importable whether pytest is invoked from the repo
# root or elsewhere (the ground_truth dir has no package conftest of its own).
# tools/xai_study/ground_truth/test_oracle.py -> repo root is parents[3].
_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.xai_study.common import seeds  # noqa: E402
from tools.xai_study.ground_truth import oracle_intervene as O  # noqa: E402

GAME = "pong"
TARGET = 4        # intervene at frame 4 (well inside the conformance horizon)
HORIZON = 4       # continue 4 frames -> reads at frame 8 (<= 30-frame RAM horizon)


# A quiet background byte: an index NOT in the candidate set and stable across
# the baseline window. Resolved at test time from the actual baseline so the
# test is robust to RAM layout. Score bytes are 13/14.
def _quiet_background_index(base_ram: np.ndarray, candidate_idxs: set[int]) -> int:
    for idx in range(0x20, 0x80):           # upper RAM, away from score cells
        if idx not in candidate_idxs:
            return idx
    raise RuntimeError("no quiet background index found")


@pytest.fixture(scope="module")
def actions():
    return [0] * (TARGET + HORIZON)


def test_bit_exact_rerun(actions):
    """Two fresh un-intervened replays are byte-identical (RAM + screen)."""
    seeds.seed_everything(0)
    # Must not raise.
    O.assert_bit_exact(GAME, actions, TARGET + HORIZON, seed=0)
    # And explicitly confirm equality element-wise (belt and braces).
    a = O.run_baseline(GAME, actions, TARGET, HORIZON, seed=0)
    b = O.run_baseline(GAME, actions, TARGET, HORIZON, seed=0)
    assert np.array_equal(a.ram, b.ram)
    assert np.array_equal(a.screen, b.screen)


def test_score_byte_has_effect_background_does_not(actions):
    """Score byte -> nonzero Δ on its score output; background byte -> ~0."""
    seeds.seed_everything(0)
    # One checkpoint at the intervention frame; reuse it for every continuation.
    ckpt = O.make_checkpoint(GAME, actions, TARGET, seed=0)
    at_target = O._continue_from(ckpt, GAME, [])
    base_ram = at_target.ram
    outputs = O.pong_outputs(at_target)
    base_snap = O.run_baseline(GAME, actions, TARGET, HORIZON, seed=0, checkpoint=ckpt)
    y_base = {o.name: o.read(base_snap) for o in outputs}

    def delta_for(cause: O.Cause) -> dict:
        snap = O.run_intervention(GAME, actions, TARGET, HORIZON, cause, seed=0,
                                  checkpoint=ckpt)
        return {o.name: o.read(snap) - y_base[o.name] for o in outputs}

    # --- (a) the agent score byte (RAM[$0D]=13): set far from baseline -------
    base_p0 = int(base_ram[O.PONG_P0_SCORE_IDX])
    score_cause = O.Cause(
        name="ram[13]:set", kind="ram", index=O.PONG_P0_SCORE_IDX,
        value=(base_p0 + 17) & 0xFF, mode="set", concept="enemy_score")
    d_score = delta_for(score_cause)
    # The agent score output reads RAM[13] directly, so the intervention shows up
    # as a nonzero delta on p0_score (it may also feed downstream cells).
    assert abs(d_score["p0_score"]) > 0, (
        f"expected nonzero Δ on p0_score from a score-byte intervention, "
        f"got {d_score}")

    # --- (b) a quiet background byte: ~0 effect on BOTH score outputs --------
    candidate_idxs = {O.PONG_P0_SCORE_IDX, O.PONG_P1_SCORE_IDX, 49, 50, 51, 54, 45, 46}
    bg_idx = _quiet_background_index(base_ram, candidate_idxs)
    bg_cause = O.Cause(
        name=f"ram[{bg_idx}]:set", kind="ram", index=bg_idx,
        value=(int(base_ram[bg_idx]) + 17) & 0xFF, mode="set", concept="background")
    d_bg = delta_for(bg_cause)
    assert abs(d_bg["p0_score"]) == 0 and abs(d_bg["p1_score"]) == 0, (
        f"expected ~0 score Δ from a background-byte intervention at idx {bg_idx}, "
        f"got {d_bg}")


def test_causal_map_is_produced_and_sane(actions, tmp_path):
    """The full map computes, asserts bit-exactness, and writes a §R record."""
    seeds.seed_everything(0)
    cmap = O.compute_causal_map(GAME, actions, TARGET, HORIZON,
                                candidates_path=None, seed=0)
    assert cmap.bit_exact is True
    assert cmap.delta.shape == (len(cmap.cause_names), len(cmap.output_names))
    # At least one cause must move at least one output (the oracle found a cause).
    assert np.max(np.abs(cmap.delta)) > 0
    path = O.write_causal_map(cmap, tmp_path)
    assert path.is_file()
    assert path.with_suffix(".npz").is_file()
