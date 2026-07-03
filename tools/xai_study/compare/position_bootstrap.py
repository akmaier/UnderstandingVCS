#!/usr/bin/env python3
"""Bootstrap-over-GAMES 95% CI for the position-regime causal-vs-gradient faithfulness
gap on the scored battery — the significance test comparable to the paper's original
6-game [-0.05, 0.32]. Per game we take the causal/intervention family mean and the
gradient/correlational family mean of the position-regime faithfulness, form the gap,
then bootstrap the mean gap over the games (seed-fixed, reproducible).

Writes tools/xai_study/compare/out/position_bootstrap.json.
Run: python3 tools/xai_study/compare/position_bootstrap.py
"""
import importlib.util as _u, json, os, random, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "position_bootstrap.json")
CAUSAL = {"causal", "intervention"}
GRADIENT = {"gradient", "correlational"}
SEED, B = 0, 20000


def _lb():
    s = _u.spec_from_file_location("lb", os.path.join(HERE, "leaderboard.py"))
    m = _u.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    lb = _lb()
    trad = {r["method"]: r.get("tradition")
            for r in json.load(open(os.path.join(HERE, "out", "leaderboard.json")))["rows"]}
    byg = defaultdict(lambda: {"c": [], "g": []})
    for _phase, _path, rec in lb.load_per_game_records():
        if lb.output_regime(rec) != "position":
            continue
        fam = trad.get(lb.method_key(rec))
        side = "c" if fam in CAUSAL else "g" if fam in GRADIENT else None
        if side is None:
            continue
        F, _S, _M = lb.triad_of(rec)
        faith = lb.clip01(F) if F is not None else lb.orient_faithfulness(
            rec.get("metric_name"), rec.get("value"))[0]
        if faith is not None:
            byg[rec.get("game")][side].append(faith)
    gaps = {g: st.mean(d["c"]) - st.mean(d["g"]) for g, d in byg.items() if d["c"] and d["g"]}
    vals = list(gaps.values())
    n = len(vals)
    mean_gap = st.mean(vals)
    rng = random.Random(SEED)
    boots = sorted(st.mean(rng.choices(vals, k=n)) for _ in range(B))
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B)]
    out = {
        "regime": "position", "method": "bootstrap over games",
        "n_games": n, "n_boot": B, "seed": SEED,
        "mean_per_game_gap": round(mean_gap, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "excludes_zero": bool(lo > 0),
        "games_used": sorted(gaps),
        "note": "causal/intervention minus gradient/correlational family mean of the "
                "position-regime faithfulness, per game, then bootstrapped over games; "
                "comparable to the paper's original 6-game position CI of [-0.05, 0.32].",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"position gap (bootstrap over {n} games): mean {mean_gap:.3f}, "
          f"95% CI [{lo:.3f}, {hi:.3f}] -> {'EXCLUDES zero' if lo > 0 else 'includes zero'}")
    print("wrote", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
