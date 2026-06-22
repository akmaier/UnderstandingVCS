#!/usr/bin/env python3
"""P2-E2-3 — offline self-check for the discovered T3 labels.

A language-independent (numpy/json) audit of the artifacts emitted by
``discover_labels.jl`` (``tools/xai_study/t3/out/discovered_<game>.json`` + the
sibling ``.npz``). It does NOT re-run jutari; it re-derives the headline numbers
straight from the committed records and asserts they are internally consistent,
so a reviewer (or CI) can validate the increment without a Julia toolchain.

Checks, per game record:
  1. SPEC §R header present (paper/phase/method/game/value/commit/oracle_ref…).
  2. Every discovered label is causal & verified, has a response_kind in
     {position, extent}, and meets the declared causal bar (|range| ≥ threshold,
     R² ≥ r2_min) recorded in ``params``.
  3. ``summary`` counts match the actual label list (n_discovered_causal,
     n_position_labels, n_extent_labels, n_novel_no_source, n_rederives_known).
  4. Novelty is consistent with the source cross-check: a ``novel`` label's
     ram_index is NOT in the union of the harvested AtariARI/OCAtari indices; a
     ``rederives_known`` label's index IS (only enforced when indices_known).
  5. The sibling .npz round-trips: label_ram_index aligns with the JSON labels
     and the sweep arrays have the declared shape.

Run:
    PY=/Users/maier/Documents/code/UnderstandingVCS/jaxtari/.venv/bin/python
    $PY tools/xai_study/t3/discover_labels_check.py        # all discovered_*.json
    $PY tools/xai_study/t3/discover_labels_check.py pong enduro

Exit 0 = all checks pass; 1 = a failure (printed).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
REQUIRED_HEADER = [
    "paper", "phase", "method", "game", "target_output", "metric_name",
    "value", "n", "where", "commit", "oracle_ref", "timestamp",
]


def _check_record(path: str) -> list[str]:
    """Return a list of failure strings ([] = pass) for one discovered_*.json."""
    fails: list[str] = []
    with open(path) as fh:
        rec = json.load(fh)
    g = rec.get("game", os.path.basename(path))

    # (1) SPEC §R header
    for k in REQUIRED_HEADER:
        if k not in rec:
            fails.append(f"[{g}] missing §R header field: {k}")

    params = rec.get("params", {})
    r2_min = params.get("r2_min", 0.70)
    move_px = params.get("move_px_min", 4.0)
    labels = rec.get("discovered_labels", [])
    summary = rec.get("summary", {})

    # (2) every label causal+verified, kind ok, meets the bar
    n_pos = n_ext = 0
    for i, l in enumerate(labels):
        tag = f"[{g}] label#{i} RAM{l.get('ram_addr_hex')}"
        if not l.get("causal") or not l.get("verified"):
            fails.append(f"{tag}: not causal/verified")
        kind = l.get("response_kind")
        if kind not in ("position", "extent"):
            fails.append(f"{tag}: bad response_kind={kind!r}")
        n_pos += kind == "position"
        n_ext += kind == "extent"
        ev = l.get("evidence", {})
        r2 = ev.get("causal_r2")
        rng = abs(ev.get("causal_range_px", 0.0))
        if r2 is None or r2 < r2_min - 1e-9:
            fails.append(f"{tag}: R²={r2} below r2_min={r2_min}")
        # position labels must clear the centroid move bar
        if kind == "position" and rng < move_px - 1e-9:
            fails.append(f"{tag}: position range={rng} below move_px={move_px}")

    # (3) summary counts match the label list
    if summary.get("n_discovered_causal") != len(labels):
        fails.append(f"[{g}] summary n_discovered_causal "
                     f"{summary.get('n_discovered_causal')} != {len(labels)}")
    if summary.get("n_position_labels") != n_pos:
        fails.append(f"[{g}] summary n_position_labels "
                     f"{summary.get('n_position_labels')} != {n_pos}")
    if summary.get("n_extent_labels") != n_ext:
        fails.append(f"[{g}] summary n_extent_labels "
                     f"{summary.get('n_extent_labels')} != {n_ext}")
    n_novel = sum(1 for l in labels if l.get("novel") is True)
    n_rederive = sum(1 for l in labels if l.get("novel") is False)
    if summary.get("n_novel_no_source") != n_novel:
        fails.append(f"[{g}] summary n_novel_no_source "
                     f"{summary.get('n_novel_no_source')} != {n_novel}")
    if summary.get("n_rederives_known") != n_rederive:
        fails.append(f"[{g}] summary n_rederives_known "
                     f"{summary.get('n_rederives_known')} != {n_rederive}")

    # (4) novelty consistent with the harvested source indices
    sc = rec.get("source_crosscheck", {})
    if sc.get("indices_known"):
        src_idx = set(sc.get("atariari_indices", [])) | set(sc.get("ocatari_indices", []))
        for l in labels:
            ci = l["ram_index"]
            if l.get("novel") is True and ci in src_idx:
                fails.append(f"[{g}] RAM idx {ci} tagged novel but IS in a source")
            if l.get("novel") is False and ci not in src_idx:
                fails.append(f"[{g}] RAM idx {ci} tagged rederives but NOT in any source")

    # (5) sibling .npz round-trips and aligns
    npz_name = rec.get("arrays")
    if npz_name:
        npz_path = os.path.join(os.path.dirname(path), npz_name)
        if not os.path.isfile(npz_path):
            fails.append(f"[{g}] declared arrays {npz_name} missing")
        else:
            z = np.load(npz_path)
            idx = z["label_ram_index"]
            if len(idx) != len(labels):
                fails.append(f"[{g}] npz label_ram_index len {len(idx)} != {len(labels)}")
            else:
                for i, l in enumerate(labels):
                    if int(idx[i]) != int(l["ram_index"]):
                        fails.append(f"[{g}] npz idx[{i}]={idx[i]} != label {l['ram_index']}")
                        break
            K = len(params.get("sweep_vals", []))
            if z["sweep_vals"].shape != (len(labels), K):
                fails.append(f"[{g}] sweep_vals shape {z['sweep_vals'].shape} "
                             f"!= ({len(labels)},{K})")
    elif labels:
        fails.append(f"[{g}] has labels but no arrays declared")

    return fails


def main(argv: list[str]) -> int:
    games = argv[1:]
    paths = ([os.path.join(OUT_DIR, f"discovered_{g}.json") for g in games]
             if games else sorted(glob.glob(os.path.join(OUT_DIR, "discovered_*.json"))))
    if not paths:
        print("[check] no discovered_*.json found in", OUT_DIR)
        return 1

    all_fails: list[str] = []
    tot_labels = tot_novel = tot_pos = tot_ext = tot_rej = 0
    for p in paths:
        if not os.path.isfile(p):
            all_fails.append(f"missing record: {p}")
            continue
        fails = _check_record(p)
        all_fails += fails
        rec = json.load(open(p))
        s = rec.get("summary", {})
        g = rec.get("game")
        tot_labels += s.get("n_discovered_causal", 0)
        tot_novel += s.get("n_novel_no_source", 0)
        tot_pos += s.get("n_position_labels", 0)
        tot_ext += s.get("n_extent_labels", 0)
        tot_rej += s.get("n_corr_only_rejected", 0)
        status = "OK" if not fails else f"FAIL ({len(fails)})"
        print(f"[check] {g:12s} labels={s.get('n_discovered_causal',0):2d} "
              f"(pos={s.get('n_position_labels',0)} ext={s.get('n_extent_labels',0)}) "
              f"novel={s.get('n_novel_no_source',0):2d} "
              f"corr_only_rejected={s.get('n_corr_only_rejected',0):4d}  {status}")

    print(f"\n[check] TOTALS across {len(paths)} games: "
          f"discovered={tot_labels} (position={tot_pos}, extent={tot_ext}) "
          f"novel={tot_novel}  correlation_only_rejected={tot_rej}")
    if all_fails:
        print("\n[check] FAILURES:")
        for f in all_fails:
            print("  -", f)
        print(f"[check] FAIL ({len(all_fails)} issues)")
        return 1
    print("[check] PASS — all discovered-label artifacts internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
