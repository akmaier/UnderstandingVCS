"""PXC1 — xitari-trace conformance harness, jaxtari side.

This file plumbs `tools/check_trace.py` into the pytest suite. It does
two things:

1. **Sanity-check the harness itself runs.** The check-trace function
   imports cleanly and the fixture files exist in the repo. This guards
   against the harness rotting.

2. **Record the bit-exact-conformance claim as `xfail`.** Running the
   harness on the bundled `pong_noop_10` fixture reveals real RAM
   divergences between jaxtari and xitari at frame 1 — that gap is
   tracked as its own work item (see PORTING_PLAN.md / STATUS.md
   "PXC1 divergences"). Marking the test `xfail` instead of skipping it
   makes the divergence visible: pytest reports it as expected-fail
   today; the day jaxtari is patched to match xitari bit-for-bit, the
   test will start *passing* and the `xfail` marker should be removed.

The harness landing (this file + `tools/check_trace.py` +
`tools/fixtures/`) is the PXC1 deliverable. Closing the bit-exact gap
is a separate, downstream effort the harness enables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tools/ is at the repo root, alongside jaxtari/. Add it to sys.path so
# the conformance harness module is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from check_trace import ConformanceError, check_trace  # noqa: E402

_PONG_ROM   = _REPO_ROOT / "xitari" / "roms" / "pong.bin"
_PONG_TRACE = _REPO_ROOT / "tools" / "fixtures" / "traces" / "pong_noop_10.jsonl"


def test_check_trace_imports():
    """Smoke: the harness module loads and exposes its entry point."""
    assert callable(check_trace)
    assert issubclass(ConformanceError, AssertionError)


def test_pong_noop_trace_fixture_exists():
    """The golden trace shipped with the harness must be in the tree —
    so contributors don't need a built xitari just to run the suite."""
    assert _PONG_TRACE.exists(), (
        f"missing fixture {_PONG_TRACE} — regenerate with "
        f"`tools/trace_dump --rom xitari/roms/pong.bin "
        f"--actions tools/fixtures/actions/pong_noop_10.txt "
        f"> tools/fixtures/traces/pong_noop_10.jsonl`"
    )


@pytest.mark.skipif(not _PONG_ROM.exists(),
                    reason="xitari pong.bin not present in this checkout")
def test_jaxtari_matches_xitari_pong_noop_10_frames():
    """The headline bit-exact claim — **NOW PASSING** as of task #64.

    Conformance chronology:
      * P3i + P4c + paddle support: 25 → 9 RAM bytes
      * Task #64 (SWCHB B/B difficulty default): 9 → **0** bytes ✓

    The last 8 bytes of the residual divergence turned out to be
    propagation from the wrong SWCHB difficulty default. xitari's
    `Switches::Switches` initializes SWCHB to 0x3F (B/B difficulty +
    COLOR + Select/Reset released); we previously initialized to
    0xFF (A/A difficulty). Pong's title-screen state branches on the
    SWCHB difficulty bits at boot, so the wrong default cascaded
    through several RAM cells. Same root cause as the visible
    Breakout paddle-size regression — Breakout uses bit 7 of SWCHB
    to toggle paddle size, so the wrong default also drew a smaller
    paddle. Both fixed by the SWCHB default change.
    """
    matched = check_trace(_PONG_ROM, _PONG_TRACE)
    assert matched == 10, f"only {matched}/10 frames matched"
