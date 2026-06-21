"""Unit tests for the xai_study.common harness (P2-E0-1).

Covers the four DoD claims:
  1. the package is importable (loader / replay / results / seeds);
  2. a results record round-trips through write -> read per SPEC §R, including
     the sibling-`.npz` array convention;
  3. seeding is reproducible;
  4. (jaxtari-gated) Pong loads, replays a few frames, and a real result record
     built from that replay round-trips.

Run with the jaxtari venv python so the emulator-backed test executes:
    jaxtari/.venv/bin/python -m pytest tools/xai_study/common/ -v
(without it, test 4 skips and 1-3 still run.)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.xai_study.common import loader, replay, results, seeds  # noqa: E402
from tools.xai_study.common.conftest import requires_jaxtari  # noqa: E402


# --- 1. importability -------------------------------------------------------
def test_package_importable():
    for mod in (loader, replay, results, seeds):
        assert mod is not None
    # public entry points exist
    assert callable(loader.load_game)
    assert callable(replay.to_frame)
    assert callable(results.write_record)
    assert callable(seeds.seed_everything)


# --- 3. seeding reproducibility (cheap; no emulator) ------------------------
def test_seed_reproducible():
    seeds.seed_everything(123)
    import random
    a = (random.random(), float(np.random.rand()))
    seeds.seed_everything(123)
    b = (random.random(), float(np.random.rand()))
    assert a == b
    assert seeds.env_seed(None) == seeds.DEFAULT_SEED
    assert seeds.env_seed(7) == 7


# --- 2. results record round-trip (SPEC §R) --------------------------------
def test_result_record_roundtrip_scalar():
    rec = results.ResultRecord(
        phase="ground_truth", method="unit_test", game="pong",
        state="f10", target_output="score_delta", metric_name="demo_metric",
        value=0.975, stderr=0.01, ci=[0.95, 0.99], n=128, seed=0,
        where="local", oracle_ref="oracle_intervene@pong#score")
    with tempfile.TemporaryDirectory() as d:
        path = results.write_record(rec, d, exp="oracle")
        assert path.name == "oracle_pong_f10.json"
        back = results.read_record(path)
    # every SPEC §R field is present
    for k in ("paper", "phase", "method", "game", "state", "target_output",
              "metric_name", "value", "stderr", "ci", "n", "seed", "where",
              "commit", "oracle_ref", "timestamp"):
        assert k in back, f"missing schema field: {k}"
    assert back["paper"] == "P2"
    assert back["value"] == 0.975
    assert back["ci"] == [0.95, 0.99]
    assert back["commit"]                     # auto-filled, non-empty
    assert back["timestamp"]                  # auto-filled
    assert back.get("arrays") in (None,)      # no arrays here


def test_result_record_roundtrip_with_arrays():
    """Sibling-.npz convention is exercised and the arrays round-trip exactly."""
    cmap = np.arange(12, dtype=np.float32).reshape(3, 4)
    rec = results.ResultRecord(
        phase="phaseB_attribution", method="integrated_gradients", game="breakout",
        target_output="paddle_pixel", metric_name="oracle_corr", value=0.42,
        n=16, seed=1, where="cluster")
    with tempfile.TemporaryDirectory() as d:
        path = results.write_record(rec, d, exp="ig", arrays={"causal_map": cmap})
        assert (Path(d) / "ig_breakout.npz").is_file()
        back = results.read_record(path)
    assert back["arrays"] == "ig_breakout.npz"
    assert "_arrays" in back
    assert np.array_equal(back["_arrays"]["causal_map"], cmap)


def test_record_stem_state_optional():
    assert results.record_stem("ig", "pong") == "ig_pong"
    assert results.record_stem("ig", "pong", "f10") == "ig_pong_f10"


# --- replay horizon guardrails (cheap; no emulator) ------------------------
def test_load_actions_from_list_and_file():
    assert replay.load_actions([0, 1, 2]) == [0, 1, 2]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# comment\n0\n\n3\n2\n")
        name = f.name
    try:
        assert replay.load_actions(name) == [0, 3, 2]
    finally:
        Path(name).unlink(missing_ok=True)


# --- 4. emulator-backed end-to-end (Pong) ----------------------------------
@requires_jaxtari
def test_pong_load_replay_and_write_record():
    # jaxtari runs eager (~3.5 s/step), so we use the faster single-boot path
    # (construction_probe=False) here: it exercises the full harness API —
    # load-by-name, deterministic replay, record round-trip — without the
    # minutes-long xitari double-boot probe. The probe is a Paper-1 conformance
    # detail, not needed to validate the harness; loader defaults to it for real
    # experiments (load_game(...) keeps construction_probe=True).
    seeds.seed_everything(0)
    env, rom = loader.load_game("pong", seed=seeds.env_seed(0),
                                construction_probe=False)
    assert rom.ndim == 1 and rom.size > 0

    n = 6
    actions = [0] * n
    snap = replay.to_frame(env, actions, frame=n)
    assert snap.frame == n
    assert snap.ram.shape == (128,)
    assert snap.ram.dtype == np.uint8
    assert snap.screen.ndim == 2 and snap.screen.shape[1] == 160

    # a fresh env replayed identically must reproduce the RAM bit-for-bit
    env2, _ = loader.load_game("pong", seed=seeds.env_seed(0),
                               construction_probe=False)
    traj = replay.trajectory(env2, actions, n=n)
    assert traj["ram"].shape == (n, 128)
    assert np.array_equal(traj["ram"][-1], snap.ram), "replay not deterministic"

    # build + round-trip a real result record from this replay
    rec = results.ResultRecord(
        phase="ground_truth", method="harness_smoke", game="pong",
        state=f"f{n}", target_output="ram_checksum",
        metric_name="ram_sum", value=int(snap.ram.sum()),
        n=n, seed=0, where="local",
        oracle_ref="harness@pong#smoke")
    with tempfile.TemporaryDirectory() as d:
        path = results.write_record(rec, d, exp="smoke",
                                    arrays={"ram": snap.ram,
                                            "screen": snap.screen})
        back = results.read_record(path)
    assert back["game"] == "pong"
    assert back["value"] == int(snap.ram.sum())
    assert np.array_equal(back["_arrays"]["ram"], snap.ram)
    assert np.array_equal(back["_arrays"]["screen"], snap.screen)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
