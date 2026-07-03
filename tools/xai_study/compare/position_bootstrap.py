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


def _gap_ci(lb, trad, regime):
    """Bootstrap-over-games 95% CI of the causal/intervention minus gradient/correlational
    FAMILY-MEAN faithfulness gap — the paper's construction: each method's faithfulness is
    the mean over games; the family mean is the mean over that family's methods; the gap is
    the family-mean difference. The CI resamples games (with replacement), recomputes every
    method's mean on the resampled games, recomputes the family means, and takes the gap.
    regime='position' uses each method's position-regime records; 'all' uses all records.
    So the point estimate equals the leaderboard family-mean gap and sits inside its own CI."""
    # per (method, game): mean faithfulness over that method's records for the game/regime
    cell = defaultdict(lambda: defaultdict(list))   # method -> game -> [faith]
    side_of = {}
    for _phase, _path, rec in lb.load_per_game_records():
        if regime == "position" and lb.output_regime(rec) != "position":
            continue
        m = lb.method_key(rec)
        fam = trad.get(m)
        side = "c" if fam in CAUSAL else "g" if fam in GRADIENT else None
        if side is None:
            continue
        F, _S, _M = lb.triad_of(rec)
        faith = lb.clip01(F) if F is not None else lb.orient_faithfulness(
            rec.get("metric_name"), rec.get("value"))[0]
        if faith is not None:
            cell[m][rec.get("game")].append(faith)
            side_of[m] = side
    method_games = {m: {g: st.mean(v) for g, v in gm.items()} for m, gm in cell.items()}
    games = sorted({g for gm in method_games.values() for g in gm})

    def family_gap(sample):
        fam_means = {"c": [], "g": []}
        for m, gm in method_games.items():
            vals = [gm[g] for g in sample if g in gm]
            if vals:
                fam_means[side_of[m]].append(st.mean(vals))
        if not fam_means["c"] or not fam_means["g"]:
            return None
        return st.mean(fam_means["c"]) - st.mean(fam_means["g"])

    point = family_gap(games)
    rng = random.Random(SEED)
    boots = sorted(x for _ in range(B) for x in [family_gap(rng.choices(games, k=len(games)))] if x is not None)
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    return {
        "regime": regime, "method": "bootstrap over games (family-mean gap)",
        "n_games": len(games), "n_boot": B, "seed": SEED,
        "family_mean_gap": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "excludes_zero": bool(lo > 0),
    }


def main():
    lb = _lb()
    trad = {r["method"]: r.get("tradition")
            for r in json.load(open(os.path.join(HERE, "out", "leaderboard.json")))["rows"]}
    pos = _gap_ci(lb, trad, "position")
    allr = _gap_ci(lb, trad, "all")
    pos["note"] = ("causal/intervention minus gradient/correlational family mean of the "
                   "position-regime faithfulness, per game, then bootstrapped over games; "
                   "comparable to the paper's original 6-game position CI of [-0.05, 0.32].")
    allr["note"] = ("same construction over ALL output regimes; comparable to the paper's "
                    "original 6-game all-regime CI of [0.232, 0.375].")
    out = {"position": pos, "all_regime": allr}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    for k, r in out.items():
        print(f"{k:10} family-mean gap (bootstrap over {r['n_games']} games): {r['family_mean_gap']:.3f}, "
              f"95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] -> "
              f"{'EXCLUDES zero' if r['excludes_zero'] else 'includes zero'}")
    print("wrote", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
