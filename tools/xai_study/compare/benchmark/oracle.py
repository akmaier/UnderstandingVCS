#!/usr/bin/env python3
"""P2-E6-2 benchmark — the ORACLE interface.

The single object every interpretability method is scored against: the §1
ground-truth causal map. For a (game, target output) the oracle exposes, per
*named cause* `u` (a RAM cell, a TIA register, a do(action) input), the EXACT
causal effect |Δy(u)| measured by intervene-then-re-run on the bit-exact
emulator (experiment_design.md §1; SPEC §E1).

How the ground truth is obtained
--------------------------------
The primary oracle generator is the Julia intervention oracle
``tools/xai_study/ground_truth/oracle_intervene.jl`` (re-run of the real ROM;
SOFT-STE forward is bit-exact to it). That generator does NOT ship in this
benchmark folder and is NOT needed to *score* a method: every Phase-B
attribution record committed under
``tools/xai_study/phaseB_attribution/out/<method>_<game>_<regime>.json``
embeds, under ``extra``:

  * ``cause_names``                 — the ordered list of candidate causes,
  * ``oracle_abs_delta_per_cause``  — {cause_name -> |Δy(u)|}, the EXACT causal
                                      importance map (the ground truth),

so this module reads the ground truth straight out of the committed records
(no ROM, no re-run). The reference set of causes + oracle deltas is identical
across methods for a given (game, regime) — they are all scored against the
same §1 oracle.

If a third party wants to *regenerate* the oracle from scratch (e.g. to add a
new game/output), they run the Julia generator on the real ROM — see
``benchmark/README.md`` §"Regenerating the oracle" and the ROM-handling note
below.

ROM non-redistribution
----------------------
Atari 2600 ROMs are NOT redistributed in this repo (they are gitignored and
used in place — SCRUM §7). Scoring a method against this benchmark needs only
the committed oracle deltas (above), so **no ROM is required to run the
benchmark**. The optional live deletion/insertion AUC (which re-runs the ROM)
references each ROM by its SHA-256 + AutoROM name via
``benchmark/rom_manifest.json``; it never ships ROM bytes.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.dirname(HERE)          # tools/xai_study/compare
STUDY = os.path.dirname(COMPARE)         # tools/xai_study
PHASEB_OUT = os.path.join(STUDY, "phaseB_attribution", "out")
GROUND_TRUTH_OUT = os.path.join(STUDY, "ground_truth", "out")

# A committed Phase-B record per (game, regime) carries the canonical oracle map.
# We prefer occlusion (a real intervention, dense oracle column) as the reference
# carrier; fall back to any method record that embeds the oracle deltas.
_PREFERRED_CARRIER = "occlusion"
_REGIME_TAGS = ("content", "position", "ball_pixel")


@dataclass
class OracleMap:
    """The §1 ground-truth causal map for one (game, regime).

    Attributes
    ----------
    game, regime : the task coordinates.
    cause_names  : ordered list of candidate causes u (RAM/TIA/joystick).
    abs_delta    : list[float] aligned to ``cause_names`` — the EXACT |Δy(u)|.
    target_output: the y the deltas are measured on (e.g. content(ram_self@54)).
    source       : the committed record path the map was read from.
    oracle_ref   : the §R ``oracle_ref`` string for provenance.
    commit       : the git commit the record was produced at.
    """
    game: str
    regime: str
    cause_names: List[str]
    abs_delta: List[float]
    target_output: str
    source: str
    oracle_ref: str = ""
    commit: str = ""
    extra: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        """{cause_name -> |Δy(u)|} convenience view."""
        return dict(zip(self.cause_names, self.abs_delta))

    def top_k(self, k: int) -> List[str]:
        """The true causal top-k causes (most important first)."""
        order = sorted(range(len(self.cause_names)),
                       key=lambda i: self.abs_delta[i], reverse=True)
        return [self.cause_names[i] for i in order[:k]]

    @property
    def n_causes(self) -> int:
        return len(self.cause_names)

    @property
    def scorable(self) -> bool:
        """A task is scorable only if its oracle column is non-degenerate: at
        least one cause has a non-zero |Δy| AND the column is not constant.
        On a constant/all-zero column, correlation and precision@k carry no
        signal (no cause matters within the conformance horizon), so even a
        perfect oracle copy would score 0 — such tasks are excluded from the
        benchmark task set (they are reported as degenerate, not failed)."""
        nz = any(v != 0.0 for v in self.abs_delta)
        non_const = len(set(self.abs_delta)) > 1
        return bool(nz and non_const)


def _carrier_records(game: str, regime: str) -> List[str]:
    """Committed records that may carry the oracle map for (game, regime),
    preferred carrier first."""
    pats = [
        os.path.join(PHASEB_OUT, f"{_PREFERRED_CARRIER}_{game}_{regime}.json"),
        os.path.join(PHASEB_OUT, f"*_{game}_{regime}.json"),
    ]
    seen: List[str] = []
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            if p not in seen:
                seen.append(p)
    return seen


def load_oracle(game: str, regime: str = "content") -> OracleMap:
    """Load the §1 ground-truth causal map for (game, regime) from the committed
    records. ``regime`` ∈ {content, position, ball_pixel}.

    Raises FileNotFoundError if no committed record embeds the oracle deltas for
    that (game, regime) — which means the oracle has not yet been produced for
    that task (regenerate via the Julia generator)."""
    for path in _carrier_records(game, regime):
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        ex = rec.get("extra") or {}
        names = ex.get("cause_names")
        deltas = ex.get("oracle_abs_delta_per_cause")
        if not names or deltas is None:
            continue
        # deltas may be a dict {name->v} or a list aligned to names
        if isinstance(deltas, dict):
            abs_delta = [float(deltas.get(n, 0.0)) for n in names]
        else:
            abs_delta = [float(v) for v in deltas]
        return OracleMap(
            game=game,
            regime=regime,
            cause_names=list(names),
            abs_delta=abs_delta,
            target_output=rec.get("target_output", ""),
            source=os.path.relpath(path, STUDY),
            oracle_ref=rec.get("oracle_ref", ""),
            commit=rec.get("commit", ""),
            extra={"output_kind": ex.get("output_kind"),
                   "content_ram_index": ex.get("content_ram_index")},
        )
    raise FileNotFoundError(
        f"no committed oracle map for game={game!r} regime={regime!r} "
        f"under {os.path.relpath(PHASEB_OUT, STUDY)} (looked for "
        f"extra.cause_names + extra.oracle_abs_delta_per_cause). "
        f"Regenerate with tools/xai_study/ground_truth/oracle_intervene.jl.")


def available_tasks(include_degenerate: bool = False) -> List[Dict[str, str]]:
    """Enumerate every (game, regime, target_output) for which a committed
    oracle map exists. By default only SCORABLE tasks (non-degenerate oracle
    column) are returned — the benchmark task set. Pass ``include_degenerate``
    to also list the degenerate (all-zero / constant) oracle columns, flagged
    ``scorable=0``."""
    tasks: List[Dict[str, str]] = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(PHASEB_OUT, "*.json"))):
        bn = os.path.basename(path)
        if bn.endswith("_core_summary.json") or bn.endswith("_all.json"):
            continue
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        ex = rec.get("extra") or {}
        if not ex.get("cause_names") or ex.get("oracle_abs_delta_per_cause") is None:
            continue
        game = rec.get("game")
        # regime token = last underscore-group matching a known tag
        regime = None
        for tag in _REGIME_TAGS:
            if bn.endswith(f"_{tag}.json"):
                regime = tag
                break
        if not game or not regime:
            continue
        key = (game, regime)
        if key in seen:
            continue
        seen.add(key)
        om = load_oracle(game, regime)
        if not om.scorable and not include_degenerate:
            continue
        tasks.append({
            "game": game,
            "regime": regime,
            "target_output": rec.get("target_output", ""),
            "n_causes": str(len(ex["cause_names"])),
            "scorable": "1" if om.scorable else "0",
        })
    tasks.sort(key=lambda t: (t["game"], t["regime"]))
    return tasks


def oracle_self_check() -> bool:
    """The oracle, scored against itself, must be perfectly faithful — a
    positive control of the interface (corr==1, p@k==1)."""
    from . import metrics  # local import to avoid cycle at module import
    ok = True
    for t in available_tasks():
        om = load_oracle(t["game"], t["regime"])
        attr = om.as_dict()  # oracle copies itself
        r = metrics.pearson_corr(attr, om)
        pak = metrics.precision_at_k(attr, om, k=3)
        if not (r > 0.999 and abs(pak - 1.0) < 1e-9):
            ok = False
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect the benchmark oracle")
    ap.add_argument("--game", default=None)
    ap.add_argument("--regime", default="content")
    a = ap.parse_args()
    if a.game:
        om = load_oracle(a.game, a.regime)
        print(f"{om.game}/{om.regime}: {om.n_causes} causes, target={om.target_output}")
        print(f"  source={om.source} commit={om.commit}")
        print(f"  true top-3: {om.top_k(3)}")
    else:
        for t in available_tasks():
            print(t)
