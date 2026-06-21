"""results — the self-describing P2 results-record writer/reader (SPEC §R).

Every P2 experiment writes one record per measurement:

    out/<phase>/<exp>_<game>[_<state>].json

with the SPEC §R fields:

    {paper, phase, method, game, frame/state, target_output, metric_name,
     value, ci/stderr, n, seed, where, commit, oracle_ref, timestamp}

Array payloads (attribution maps, causal maps, per-frame tensors, ...) do NOT go
in the JSON; they go in a **sibling `.npz`** with the same stem, and the JSON
records that filename under `arrays` so the record stays self-describing and the
JSON stays small/diffable. `commit` and `timestamp` are filled automatically.

This module emits **per-record JSON only**. The append-only `results_index.csv`
is SM-merged at Review (SPEC §R) — do not write it here.

Usage:
    from tools.xai_study.common import results
    rec = results.ResultRecord(
        phase="ground_truth", method="intervention_oracle", game="pong",
        state="f10", target_output="score_delta", metric_name="causal_corr",
        value=0.97, n=128, seed=0, where="local",
        oracle_ref="oracle_intervene@pong#score")
    path = results.write_record(rec, out_dir, exp="oracle",
                                arrays={"causal_map": cmap})
    back = results.read_record(path)        # dict, with arrays loaded lazily
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

#: Constant for this paper (SPEC §R `paper`).
PAPER = "P2"

#: SPEC §R fields, in canonical order, for stable JSON / index columns.
SCHEMA_FIELDS = (
    "paper", "phase", "method", "game", "state", "target_output",
    "metric_name", "value", "stderr", "ci", "n", "seed", "where",
    "commit", "oracle_ref", "timestamp", "arrays", "extra",
)


def git_commit(repo: Optional[Path] = None) -> str:
    """Return the current short git commit hash, or 'unknown' if unavailable."""
    root = Path(repo) if repo else Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ResultRecord:
    """One self-describing measurement (SPEC §R).

    Required: `phase, method, game, target_output, metric_name, value`.
    `state` is the SPEC's `frame/state` slot (e.g. ``"f10"`` or ``"start"``);
    `None` if the measurement is not frame-specific. `commit`/`timestamp` are
    filled by `to_dict()`/`write_record` if left empty.
    """
    phase: str
    method: str
    game: str
    target_output: str
    metric_name: str
    value: Any
    state: Optional[str] = None
    stderr: Optional[float] = None
    ci: Optional[Any] = None          # e.g. [lo, hi]
    n: Optional[int] = None
    seed: Optional[int] = None
    where: str = "local"              # local | cluster
    oracle_ref: Optional[str] = None
    commit: Optional[str] = None
    timestamp: Optional[str] = None
    paper: str = PAPER
    arrays: Optional[str] = None      # sibling .npz filename, set by write_record
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict in `SCHEMA_FIELDS` order, auto-filling
        commit/timestamp. NumPy scalars are coerced to native Python."""
        d = {
            "paper": self.paper,
            "phase": self.phase,
            "method": self.method,
            "game": self.game,
            "state": self.state,
            "target_output": self.target_output,
            "metric_name": self.metric_name,
            "value": _to_native(self.value),
            "stderr": _to_native(self.stderr),
            "ci": _to_native(self.ci),
            "n": self.n,
            "seed": self.seed,
            "where": self.where,
            "commit": self.commit or git_commit(),
            "oracle_ref": self.oracle_ref,
            "timestamp": self.timestamp or utc_timestamp(),
            "arrays": self.arrays,
            "extra": _to_native(self.extra),
        }
        return {k: d[k] for k in SCHEMA_FIELDS}


def _to_native(x: Any) -> Any:
    """Recursively convert NumPy types to JSON-serializable Python types."""
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def record_stem(exp: str, game: str, state: Optional[str] = None) -> str:
    """Build the filename stem `<exp>_<game>[_<state>]` per SPEC §R."""
    stem = f"{exp}_{game}"
    if state:
        stem += f"_{state}"
    return stem


def write_record(record: ResultRecord,
                 out_dir: os.PathLike,
                 exp: str,
                 arrays: Optional[Mapping[str, np.ndarray]] = None) -> Path:
    """Write `record` to `<out_dir>/<exp>_<game>[_<state>].json` (+ sibling .npz).

    `out_dir` is the experiment's `out/` directory (created if needed); SPEC §R's
    `out/<phase>/...` layout is the caller's choice of `out_dir`. If `arrays` is
    given, they are saved to a sibling `.npz` (same stem) and the record's
    `arrays` field is set to that filename. Returns the JSON path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = record_stem(exp, record.game, record.state)
    json_path = out / f"{stem}.json"
    if arrays:
        npz_path = out / f"{stem}.npz"
        np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in arrays.items()})
        record.arrays = npz_path.name
    payload = record.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return json_path


def read_record(json_path: os.PathLike, load_arrays: bool = True) -> Dict[str, Any]:
    """Read a record JSON back; if it names a sibling `.npz`, load it under
    `"_arrays"` as a dict of NumPy arrays. Round-trips `write_record`."""
    p = Path(json_path)
    data = json.loads(p.read_text())
    if load_arrays and data.get("arrays"):
        npz_path = p.parent / data["arrays"]
        if npz_path.is_file():
            with np.load(npz_path) as z:
                data["_arrays"] = {k: z[k] for k in z.files}
    return data


# Re-export os for the type hint above without a top-level import cycle.
import os  # noqa: E402
