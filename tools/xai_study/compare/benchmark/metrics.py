#!/usr/bin/env python3
"""P2-E6-2 benchmark — the METRIC definitions.

The faithfulness metrics every method is scored on against the §1 oracle
(experiment_design.md §5; §0 correctness triad). These reproduce, in Python,
the EXACT formulas used by the committed Julia scorers
(``tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl`` ::
``pearson`` / ``spearman`` / ``_rank`` / ``precision_at_k`` /
``deletion_insertion_auc``), so a method scored through this benchmark gets a
number directly comparable to the leaderboard (E6-1).

A method's output is an ATTRIBUTION over the oracle's named causes:
``{cause_name -> score}`` (higher = more important). It is aligned to the
oracle's ``cause_names`` order before any metric is computed; missing causes
score 0; unknown extra causes are ignored.

Offline metrics (no ROM, computed from the committed oracle deltas):
  * pearson_corr / spearman_corr   — corr(attr, |Δy|)            ∈ [-1,1]
  * precision_at_k                 — |top-k(attr) ∩ top-k(oracle)|/k ∈ [0,1]
  * faithfulness                   — corr clipped to [0,1] (the F leg, §0)

Live metrics (need the ROM; optional, via a re-run hook — see ``run.py`` and
the README): deletion_auc / insertion_auc. The example runs offline, so these
are reported as ``None`` unless a ``rerun_fn`` is supplied.

The correctness triad (F / S / M, §0):
  * F (Faithful)   = faithfulness (corr-with-oracle clipped to [0,1]) — measured
                     here from the attribution.
  * S (Sufficient) = predicts y under a HELD-OUT intervention (needs the oracle's
                     held-out column / a ROM re-run). A generic method supplies it
                     via ``method_output['S']`` if it can; else None.
  * M (Minimal)    = parsimony / right-level. Supplied by the method via
                     ``method_output['M']`` (e.g. 1 − over-claim rate) if it can;
                     else None.
F is always benchmark-measured; S and M are method-reported (the benchmark
records whichever legs are available, mirroring the committed records where 49
carry an ``extra.triad`` block and the rest carry F only).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def clip01(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return max(0.0, min(1.0, float(x)))


def _align(attr: Dict[str, float], cause_names: Sequence[str]) -> List[float]:
    """Project a {cause->score} attribution onto the oracle's cause order."""
    return [float(attr.get(n, 0.0)) for n in cause_names]


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson r; 0 when either side is constant (matches the Julia scorer)."""
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0.0 or vb == 0.0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def _rank(v: Sequence[float]) -> List[float]:
    """Average ranks (ties → mean rank), mirroring the Julia ``_rank``."""
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 1-based ranks, like Julia
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson_corr(attr: Dict[str, float], oracle) -> float:
    """corr(attr, |Δy|) over the oracle's causes (∈ [-1,1])."""
    a = _align(attr, oracle.cause_names)
    return _pearson(a, oracle.abs_delta)


def spearman_corr(attr: Dict[str, float], oracle) -> float:
    a = _align(attr, oracle.cause_names)
    return _pearson(_rank(a), _rank(oracle.abs_delta))


def precision_at_k(attr: Dict[str, float], oracle, k: int = 3) -> float:
    """|top-k(attr) ∩ true top-k| / k."""
    names = oracle.cause_names
    a = _align(attr, names)
    k = min(k, len(names))
    if k <= 0:
        return 0.0
    a_top = set(sorted(range(len(names)), key=lambda i: a[i], reverse=True)[:k])
    o_top = set(sorted(range(len(names)),
                       key=lambda i: oracle.abs_delta[i], reverse=True)[:k])
    return len(a_top & o_top) / k


def faithfulness(attr: Dict[str, float], oracle) -> float:
    """The F leg (§0): corr-with-oracle clipped to [0,1] (an anti-correlated map
    is no more faithful than a random one)."""
    return max(0.0, pearson_corr(attr, oracle))


def deletion_insertion_auc(attr: Dict[str, float], oracle,
                           rerun_fn=None) -> Tuple[Optional[float], Optional[float]]:
    """Deletion / insertion AUC measured ON THE TRUE VCS (needs the ROM).

    ``rerun_fn(order_of_cause_names, mode) -> normalised_curve`` is an optional
    hook that re-runs the real emulator, occluding (mode='deletion') or adding
    back (mode='insertion') causes in the supplied order and returning the
    normalised y-curve; the AUC is its trapezoidal mean on a unit grid (matching
    the Julia ``deletion_insertion_auc``). With no hook (offline / no ROM) both
    are None. The benchmark is fully runnable offline without these."""
    if rerun_fn is None:
        return None, None
    a = _align(attr, oracle.cause_names)
    order = sorted(range(len(oracle.cause_names)), key=lambda i: a[i], reverse=True)
    order_names = [oracle.cause_names[i] for i in order]
    del_curve = rerun_fn(order_names, "deletion")
    ins_curve = rerun_fn(order_names, "insertion")
    return _trapz_unit(del_curve), _trapz_unit(ins_curve)


def _trapz_unit(curve: Sequence[float]) -> float:
    """Trapezoidal mean of a curve on a unit [0,1] grid (AUC convention)."""
    n = len(curve)
    if n < 2:
        return float(curve[0]) if curve else 0.0
    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (curve[i] + curve[i + 1])
    return s / (n - 1)


def score_method(method_output: Dict, oracle, k: int = 3,
                 rerun_fn=None) -> Dict:
    """Score one method's output against one oracle map. Returns the §R-shaped
    metric block.

    ``method_output`` must contain ``attribution: {cause_name -> score}`` and
    may optionally contain ``S`` and ``M`` (the held-out-sufficiency and
    minimality legs the method itself reports; the benchmark cannot synthesise
    them generically)."""
    attr = method_output.get("attribution") or {}
    if not isinstance(attr, dict):
        raise TypeError("method_output['attribution'] must be {cause_name: score}")
    pr = pearson_corr(attr, oracle)
    sp = spearman_corr(attr, oracle)
    pak = precision_at_k(attr, oracle, k)
    F = clip01(max(0.0, pr))
    S = clip01(method_output.get("S"))
    M = clip01(method_output.get("M"))
    del_auc, ins_auc = deletion_insertion_auc(attr, oracle, rerun_fn=rerun_fn)
    legs = [x for x in (F, S, M) if x is not None]
    if len(legs) == 3:
        fsm = (legs[0] * legs[1] * legs[2]) ** (1.0 / 3.0)
    elif legs:
        fsm = sum(legs) / len(legs)
    else:
        fsm = None
    return {
        "pearson_corr": round(pr, 6),
        "spearman_corr": round(sp, 6),
        "precision_at_k": round(pak, 6),
        "topk": k,
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "faithfulness": None if F is None else round(F, 6),
        "triad": {
            "F": None if F is None else round(F, 6),
            "S": None if S is None else round(S, 6),
            "M": None if M is None else round(M, 6),
        },
        "FSM_composite": None if fsm is None else round(fsm, 6),
        "n_causes": oracle.n_causes,
        "true_top_k": oracle.top_k(k),
    }
