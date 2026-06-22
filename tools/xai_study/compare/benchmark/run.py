#!/usr/bin/env python3
"""P2-E6-2 benchmark — the runnable ENTRY POINT.

Score one interpretability method end-to-end against the P2 ground-truth oracle
and emit a §R-schema record per task. This is the single documented command a
third party runs.

Usage
-----
    # self-test: run the bundled positive control + baselines over all tasks
    python -m tools.xai_study.compare.benchmark.run --self-test

    # score one bundled example on one game
    python -m tools.xai_study.compare.benchmark.run --method oracle_copy --game pong

    # score every bundled example over every task, write records
    python -m tools.xai_study.compare.benchmark.run --method magnitude_proxy

    # plug in YOUR method (a callable method(task, oracle) -> {attribution,S,M})
    python -m tools.xai_study.compare.benchmark.run --method my_pkg.my_mod:my_method

Records are written to ``tools/xai_study/compare/benchmark/out/<method>_<game>_<regime>.json``
in the SPEC §R schema, so the E6-1 leaderboard can read them like any other
phase record (faithfulness from ``extra.triad.F``).

No ROM is needed: the oracle ground truth is read from the committed records
(oracle.py). The optional live deletion/insertion AUC (which re-runs the ROM) is
off by default; see ``--with-rerun`` and the README.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List

# allow both `python -m tools.xai_study.compare.benchmark.run` and direct exec
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    from tools.xai_study.compare.benchmark import metrics, oracle as oracle_mod, tasks as tasks_mod
    from tools.xai_study.compare.benchmark.example_method import EXAMPLES
else:
    from . import metrics, oracle as oracle_mod, tasks as tasks_mod
    from .example_method import EXAMPLES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def resolve_method(spec: str) -> Callable:
    """Resolve a method by short name (bundled example) or 'module:callable'."""
    if spec in EXAMPLES:
        return EXAMPLES[spec]
    if ":" in spec:
        mod_name, fn_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)
    raise ValueError(f"unknown method {spec!r}; bundled: {sorted(EXAMPLES)} "
                     f"or pass 'module:callable'")


def score_task(method: Callable, method_name: str, task, rerun_fn=None) -> Dict:
    """Run + score one method on one task, return the §R record dict."""
    om = oracle_mod.load_oracle(task.game, task.regime)
    out = method(task, om)
    block = metrics.score_method(out, om, k=3, rerun_fn=rerun_fn)
    record = {
        "paper": "P2",
        "phase": "benchmark",
        "method": method_name,
        "game": task.game,
        "frame": None,
        "state": "f120+30",
        "target_output": task.target_output,
        "metric_name": "pearson_corr_with_oracle",
        "value": block["pearson_corr"],
        "ci": None,
        "stderr": None,
        "n": om.n_causes,
        "seed": 0,
        "where": "local",
        "commit": _git_commit(),
        "oracle_ref": om.oracle_ref or f"oracle@{task.game}#{task.regime}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra": {
            "benchmark": "P2-E6-2",
            "regime": task.regime,
            "metrics": {k: block[k] for k in
                        ("pearson_corr", "spearman_corr", "precision_at_k",
                         "topk", "deletion_auc", "insertion_auc")},
            "triad": block["triad"],
            "FSM_composite": block["FSM_composite"],
            "faithfulness": block["faithfulness"],
            "cause_names": om.cause_names,
            "true_top_k": block["true_top_k"],
            "oracle_source": om.source,
            "oracle_commit": om.commit,
        },
    }
    return record


def run(method_spec: str, games: List[str], regimes: List[str],
        write: bool = True, rerun_fn=None, quiet: bool = False) -> List[Dict]:
    method = resolve_method(method_spec)
    method_name = method_spec.split(":")[-1] if ":" in method_spec else method_spec
    all_t = tasks_mod.all_tasks()
    sel = [t for t in all_t
           if (not games or t.game in games) and (not regimes or t.regime in regimes)]
    if not sel:
        raise SystemExit(f"no tasks for games={games} regimes={regimes}; "
                         f"available games={sorted({t.game for t in all_t})} "
                         f"regimes={sorted({t.regime for t in all_t})}")
    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for t in sel:
        rec = score_task(method, method_name, t, rerun_fn=rerun_fn)
        records.append(rec)
        if write:
            p = os.path.join(OUT_DIR, f"{method_name}_{t.game}_{t.regime}.json")
            with open(p, "w") as f:
                json.dump(rec, f, indent=2)
        if not quiet:
            m = rec["extra"]["metrics"]
            print(f"  {method_name:<16} {t.id:<26} "
                  f"corr={m['pearson_corr']:+.3f} p@3={m['precision_at_k']:.3f} "
                  f"F={rec['extra']['faithfulness']}")
    return records


def self_test() -> int:
    """End-to-end self-check: the bundled examples score as expected.

    (1) the oracle-copy positive control is perfectly faithful on every task;
    (2) the uniform baseline is at the floor (corr 0);
    (3) a real (oracle-free) method produces a finite faithfulness score on Pong;
    (4) the oracle interface self-check passes (oracle-vs-itself corr==1)."""
    print("[benchmark self-test] scoring bundled methods over all tasks ...")
    checks = []

    oc = run("oracle_copy", [], [], write=False, quiet=True)
    oc_ok = all(abs(r["extra"]["faithfulness"] - 1.0) < 1e-6
                and abs(r["extra"]["metrics"]["precision_at_k"] - 1.0) < 1e-9
                for r in oc)
    checks.append(("oracle_copy is perfectly faithful on every task", oc_ok,
                   f"n_tasks={len(oc)} "
                   f"min_F={min(r['extra']['faithfulness'] for r in oc):.4f}"))

    uni = run("uniform", [], [], write=False, quiet=True)
    uni_ok = all(abs(r["extra"]["metrics"]["pearson_corr"]) < 1e-9 for r in uni)
    checks.append(("uniform baseline is at the floor (corr 0)", uni_ok,
                   f"max|corr|={max(abs(r['extra']['metrics']['pearson_corr']) for r in uni):.2e}"))

    # (3) a genuine end-to-end score for a non-oracle method on Pong
    mp = run("magnitude_proxy", ["pong"], ["content"], write=False, quiet=True)
    mp_ok = (len(mp) == 1
             and mp[0]["extra"]["faithfulness"] is not None
             and 0.0 <= mp[0]["extra"]["faithfulness"] <= 1.0)
    faith_pong = mp[0]["extra"]["faithfulness"] if mp else None
    checks.append(("magnitude_proxy produces a finite faithfulness on pong/content",
                   mp_ok, f"faithfulness={faith_pong}"))

    # (4) the oracle interface positive control
    iface_ok = oracle_mod.oracle_self_check()
    checks.append(("oracle interface self-check (oracle-vs-itself corr==1)",
                   iface_ok, "ok" if iface_ok else "FAILED"))

    n_tasks = len(tasks_mod.all_tasks())
    checks.append(("benchmark exposes >= 6 tasks", n_tasks >= 6,
                   f"n_tasks={n_tasks}"))

    print("\n[benchmark self-test] results:")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    passed = all(ok for _, ok, _ in checks)
    print(f"\nSELF-CHECK: {'PASS' if passed else 'FAIL'}  "
          f"(headline example faithfulness on pong/content = {faith_pong})")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=None,
                    help="bundled example name (%s) or 'module:callable'"
                         % ", ".join(sorted(EXAMPLES)))
    ap.add_argument("--game", default=None,
                    help="restrict to one game (default: all core games)")
    ap.add_argument("--regime", default=None,
                    help="restrict to one regime: content|position|ball_pixel")
    ap.add_argument("--no-write", action="store_true",
                    help="score but do not write records to out/")
    ap.add_argument("--self-test", action="store_true",
                    help="run the bundled end-to-end self-check and exit")
    ap.add_argument("--list-tasks", action="store_true",
                    help="print the benchmark task set and exit")
    a = ap.parse_args()

    if a.list_tasks:
        for t in tasks_mod.all_tasks():
            print(f"{t.id:<28} target={t.target_output:<26} n_causes={t.n_causes}")
        return 0
    if a.self_test:
        return self_test()
    if not a.method:
        ap.error("one of --method / --self-test / --list-tasks is required")

    games = [a.game] if a.game else []
    regimes = [a.regime] if a.regime else []
    print(f"[benchmark] scoring method={a.method} "
          f"games={games or 'ALL'} regimes={regimes or 'ALL'}")
    recs = run(a.method, games, regimes, write=not a.no_write)
    if recs:
        import statistics
        fs = [r["extra"]["faithfulness"] for r in recs
              if r["extra"]["faithfulness"] is not None]
        mean_f = statistics.mean(fs) if fs else None
        print(f"[benchmark] {len(recs)} records, mean faithfulness = "
              f"{round(mean_f, 4) if mean_f is not None else None}"
              + ("" if a.no_write else f"  -> {os.path.relpath(OUT_DIR)}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
