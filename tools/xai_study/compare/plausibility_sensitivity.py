#!/usr/bin/env python3
"""Plausibility-proxy sensitivity analysis (P2 R6-fix; makes §3.4 / Supplement S7 true).

The leaderboard carries two reporting axes: faithfulness against the §3.2 intervention
oracle (measured) and a *plausibility proxy* (a documented, tradition-level value, NOT a
human-subjects measurement). §3.4 of the Methods promises (a) that the proxy assignment and
its sources are tabulated in the Supplement, and (b) a sensitivity analysis confirming the
headline contrast survives alternative proxy values. This script delivers (b): it computes
the headline faithfulness-vs-proxy relationship from the committed records, then perturbs the
proxy under several stated schemes and reports whether the headline contrast (plausible !=
faithful; a low-or-negative association between proxy and faithfulness; a populated
"danger zone" of high-proxy / low-faithfulness methods) survives.

It is fully deterministic. There is NO randomness: every "jitter" is a fixed offset keyed by
the method's stable index (a vary-the-row-by-index scheme), so the result re-derives
bit-for-bit on any machine. Run with the repo's jaxtari venv:

    jaxtari/.venv/bin/python tools/xai_study/compare/plausibility_sensitivity.py

Reads:
    tools/xai_study/compare/out/leaderboard.json   (rows: method, tradition,
                                                     faithfulness, plausibility_proxy;
                                                     danger_zone_discrepancy_table;
                                                     headline_contrast)
    tools/xai_study/compare/out/leaderboard_ci.csv  (cross-check of method F means)
    tools/xai_study/compare/out/faithful_demo.json  (the per-method anchor:
                                                     saliency proxy 0.9 / F 0.0 vs
                                                     activation patching proxy 0.5 / F 1.0)
Emits:
    tools/xai_study/compare/out/plausibility_sensitivity.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LEADERBOARD = os.path.join(OUT, "leaderboard.json")
LEADERBOARD_CI = os.path.join(OUT, "leaderboard_ci.csv")
FAITHFUL_DEMO = os.path.join(OUT, "faithful_demo.json")
SENS_CSV = os.path.join(OUT, "plausibility_sensitivity.csv")

# Danger zone: an explanation is "convincing yet wrong" when its tradition-level
# plausibility proxy is high but its measured faithfulness is low.
DANGER_PROXY_HI = 0.70   # proxy at or above this counts as "high plausibility"
DANGER_FAITH_LO = 0.40   # faithfulness below this counts as "low faithfulness"


# ---------------------------------------------------------------------------
# Rank-correlation helpers (no scipy dependency; spearman on the method rows).
# ---------------------------------------------------------------------------
def _ranks(xs):
    """Average ranks (1-based), ties shared, for a list of values."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of positions i..j, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


# ---------------------------------------------------------------------------
# Load.
# ---------------------------------------------------------------------------
def load_methods():
    with open(LEADERBOARD) as f:
        lb = json.load(f)
    methods = []
    for r in lb["rows"]:
        if r.get("is_positive_control"):
            continue  # the oracle is the positive control, not a method under test
        methods.append(
            {
                "method": r["method"],
                "tradition": r["tradition"],
                "faithfulness": float(r["faithfulness"]),
                "plausibility_proxy": float(r["plausibility_proxy"]),
            }
        )
    # stable, documented order: phase order in the leaderboard, then by method name index
    methods.sort(key=lambda m: m["method"])
    return lb, methods


# ---------------------------------------------------------------------------
# Headline statistics for a given proxy vector.
# ---------------------------------------------------------------------------
def headline_stats(methods, proxy):
    """Return the headline relationship for one proxy assignment.

    rho             : Spearman rank correlation, faithfulness vs proxy (we EXPECT it
                      to be low or negative: plausible != faithful).
    n_danger        : count of high-proxy / low-faithfulness "danger zone" methods.
    anchor_gap      : (proxy_saliency - proxy_actpatch); the headline per-method anchor
                      is saliency proxy 0.9 / F 0.0 vs activation patching proxy 0.5 / F 1.0,
                      i.e. the *less* faithful method carries the *higher* proxy. We report
                      whether that sign survives (anchor_gap > 0 with F ordering reversed).
    danger_methods  : sorted list of danger-zone method names.
    """
    F = [m["faithfulness"] for m in methods]
    rho = spearman(F, proxy)
    danger = [
        m["method"]
        for m, p in zip(methods, proxy)
        if p >= DANGER_PROXY_HI and m["faithfulness"] < DANGER_FAITH_LO
    ]
    by_name = {m["method"]: (m["faithfulness"], p) for m, p in zip(methods, proxy)}
    sal = by_name.get("vanilla_saliency")
    ap = by_name.get("activation_patching")
    anchor_gap = None
    anchor_holds = None
    if sal is not None and ap is not None:
        anchor_gap = sal[1] - ap[1]  # proxy difference
        # anchor holds when the popular (saliency) is MORE plausible yet LESS faithful
        anchor_holds = (anchor_gap > 0.0) and (sal[0] < ap[0])
    return {
        "rho": rho,
        "n_danger": len(danger),
        "danger_methods": sorted(danger),
        "anchor_gap": anchor_gap,
        "anchor_holds": anchor_holds,
    }


# ---------------------------------------------------------------------------
# Deterministic perturbation schemes (NO random numbers).
# ---------------------------------------------------------------------------
def signed_index_offset(i, sigma):
    """Deterministic fixed offset for row i with magnitude sigma.

    A vary-by-index scheme: alternate sign by parity and scale by a fixed fractional
    pattern of sigma, so every row gets a distinct, reproducible nudge in [-sigma, +sigma].
    """
    # fractional magnitudes cycle 1.0, 0.5, 0.75, 0.25 of sigma; sign alternates with i
    mags = (1.0, 0.5, 0.75, 0.25)
    sign = 1.0 if (i % 2 == 0) else -1.0
    return sign * mags[i % len(mags)] * sigma


def clamp01(x):
    return max(0.0, min(1.0, x))


def scheme_baseline(methods):
    return [m["plausibility_proxy"] for m in methods]


def scheme_jitter(methods, sigma):
    """Additive deterministic jitter of magnitude sigma, keyed by row index."""
    return [
        clamp01(m["plausibility_proxy"] + signed_index_offset(i, sigma))
        for i, m in enumerate(methods)
    ]


def scheme_rank_preserving(methods, sigma):
    """Jitter that keeps the tradition ordering: every proxy nudged in the SAME direction
    (compress toward the mean by sigma), so the rank order of proxies is preserved."""
    mean = sum(m["plausibility_proxy"] for m in methods) / len(methods)
    return [
        clamp01(m["plausibility_proxy"] - sigma * (m["plausibility_proxy"] - mean))
        for m in methods
    ]


def scheme_rank_jittering(methods, sigma):
    """Jitter large enough to swap neighbouring proxy ranks (the index-keyed offset at
    sigma can reorder methods whose proxies are within 2*sigma)."""
    return scheme_jitter(methods, sigma)  # same mechanism, reported at the larger sigmas


def scheme_leave_one_tradition_out(methods, tradition):
    """Drop one whole tradition and recompute on the remainder."""
    keep = [m for m in methods if m["tradition"] != tradition]
    proxy = [m["plausibility_proxy"] for m in keep]
    return keep, proxy


# ---------------------------------------------------------------------------
# Cross-checks against the committed anchors.
# ---------------------------------------------------------------------------
def cross_check(lb, methods):
    notes = []
    # 1. faithful_demo anchor: saliency 0.9/F0.0 vs activation_patching 0.5/F1.0
    with open(FAITHFUL_DEMO) as f:
        demo = json.load(f)
    pc = demo["pair_contrast"]
    by = {m["method"]: m for m in methods}
    assert abs(by["vanilla_saliency"]["plausibility_proxy"] - 0.9) < 1e-9, \
        "saliency proxy anchor != 0.9"
    assert abs(by["activation_patching"]["plausibility_proxy"] - 0.5) < 1e-9, \
        "activation_patching proxy anchor != 0.5"
    assert abs(by["activation_patching"]["faithfulness"] - 1.0) < 1e-9
    # The popular gradient tool sits near chance on the position regime and is far below
    # the faithful causal method. (Was hard-coded == 0.0, which encoded the OLD aggregation
    # bug where the popular method's ONLY position record was a stray pilot at exactly 0.0;
    # with the position bucket now correctly populated from every game's records the measured
    # per-cause faithfulness is a small positive near-chance number, not identically zero.)
    pop_pos = pc["popular_faithfulness_position"]
    faith_pos = pc["faithful_faithfulness_position"]
    assert pop_pos <= 0.30, f"popular position faithfulness not near chance: {pop_pos}"
    assert faith_pos - pop_pos > 0.5, \
        f"faithful does not clear popular on position by a wide margin: {faith_pos} vs {pop_pos}"
    notes.append(f"anchor OK: popular proxy 0.9 / Fpos {pop_pos:.3f} (near chance) vs "
                 f"act.patch proxy 0.5 / F 1.0 / Fpos {faith_pos:.3f}")
    # 2. leaderboard_ci.csv F means agree with leaderboard.json faithfulness (method rows)
    ci = {}
    with open(LEADERBOARD_CI) as f:
        for row in csv.DictReader(f):
            if row["kind"] == "method":
                ci[row["method"]] = float(row["mean_over_records"])
    mism = []
    for m in methods:
        if m["method"] in ci and abs(ci[m["method"]] - m["faithfulness"]) > 0.02:
            mism.append((m["method"], ci[m["method"]], m["faithfulness"]))
    if mism:
        notes.append("WARN ci/leaderboard F mismatch >0.02: " + str(mism))
    else:
        notes.append("leaderboard_ci.csv F means agree with leaderboard.json (<=0.02)")
    return notes


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    lb, methods = load_methods()
    notes = cross_check(lb, methods)

    base_proxy = scheme_baseline(methods)
    base = headline_stats(methods, base_proxy)

    rows = []

    def add(scheme, param, m_set, proxy):
        s = headline_stats(m_set, proxy)
        rows.append(
            {
                "scheme": scheme,
                "param": param,
                "n_methods": len(m_set),
                "spearman_F_vs_proxy": round(s["rho"], 4),
                "n_danger_zone": s["n_danger"],
                "anchor_gap_proxy": (
                    "" if s["anchor_gap"] is None else round(s["anchor_gap"], 4)
                ),
                "anchor_holds": (
                    "" if s["anchor_holds"] is None else int(s["anchor_holds"])
                ),
                "danger_methods": ";".join(s["danger_methods"]),
            }
        )
        return s

    add("baseline", "", methods, base_proxy)

    # additive deterministic jitter, rank-jittering at the larger sigmas
    for sigma in (0.05, 0.10, 0.20):
        add("jitter_additive", sigma, methods, scheme_jitter(methods, sigma))

    # rank-preserving compression toward the mean
    for sigma in (0.10, 0.20):
        add("rank_preserving", sigma, methods, scheme_rank_preserving(methods, sigma))

    # rank-jittering (same index-keyed offsets, reported at the magnitudes that can
    # reorder neighbouring proxies)
    for sigma in (0.10, 0.20):
        add("rank_jittering", sigma, methods, scheme_rank_jittering(methods, sigma))

    # leave-one-tradition-out
    for tradition in sorted({m["tradition"] for m in methods}):
        m_set, proxy = scheme_leave_one_tradition_out(methods, tradition)
        add("leave_one_tradition_out", tradition, m_set, proxy)

    # write csv
    fields = [
        "scheme",
        "param",
        "n_methods",
        "spearman_F_vs_proxy",
        "n_danger_zone",
        "anchor_gap_proxy",
        "anchor_holds",
        "danger_methods",
    ]
    with open(SENS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- self-check: the headline contrast must SURVIVE every perturbation ----
    # Contrast survives when, across all schemes:
    #   (i)   the F-vs-proxy association is never strongly positive (rho <= 0.10), and
    #   (ii)  the danger zone stays populated (n_danger >= a sturdy floor), and
    #   (iii) the per-method anchor sign holds wherever both anchors are present.
    rhos = [r["spearman_F_vs_proxy"] for r in rows]
    dangers = [r["n_danger_zone"] for r in rows]
    anchors = [r["anchor_holds"] for r in rows if r["anchor_holds"] != ""]
    survive_rho = max(rhos) <= 0.10
    survive_danger = min(dangers) >= 6
    survive_anchor = all(a == 1 for a in anchors)
    survived = survive_rho and survive_danger and survive_anchor

    print("Plausibility-proxy sensitivity (deterministic; no RNG)")
    print("-" * 64)
    print(f"baseline: spearman(F,proxy) = {base['rho']:+.4f}  "
          f"danger-zone methods = {base['n_danger']}")
    print(f"baseline danger zone: {', '.join(base['danger_methods'])}")
    print()
    print(f"{'scheme':<24}{'param':>8}{'n':>4}{'rho':>9}{'danger':>8}{'anchor':>8}")
    for r in rows:
        print(f"{r['scheme']:<24}{str(r['param']):>8}{r['n_methods']:>4}"
              f"{r['spearman_F_vs_proxy']:>9}{r['n_danger_zone']:>8}"
              f"{str(r['anchor_holds']):>8}")
    print()
    for n in notes:
        print("note:", n)
    print()
    print(f"rho range over all schemes: [{min(rhos):+.4f}, {max(rhos):+.4f}]")
    print(f"danger-zone count range:    [{min(dangers)}, {max(dangers)}]")
    print(f"anchor sign holds:          {survive_anchor} "
          f"({sum(1 for a in anchors if a==1)}/{len(anchors)} schemes with both anchors)")
    print()
    print(f"SELF-CHECK survive_rho(<=0.10)={survive_rho}  "
          f"survive_danger(>=6)={survive_danger}  survive_anchor={survive_anchor}")
    print(f"HEADLINE CONTRAST SURVIVES = {survived}")
    print(f"\nwrote {os.path.relpath(SENS_CSV, os.path.dirname(HERE))}")

    if not survived:
        print("FAIL: headline contrast did NOT survive every perturbation", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
