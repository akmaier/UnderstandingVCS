#!/usr/bin/env python3
"""Generate docs/site_data.json — the SINGLE source of Paper-2 result numbers.

This is the only place the results-audit site gets its P2 faithfulness numbers.
build_pages.py reads docs/site_data.json (and nothing else) for every P2 number,
so no two pages can ever show a conflicting value. Nothing here is transcribed by
hand: every number is read from a committed analysis output.

Sources (all committed):
  - tools/xai_study/compare/leaderboard.py loaders — per-(method,game) faithfulness
    computed the same way the leaderboard aggregates it (triad.F where present,
    else the record's value re-oriented via FAITHFULNESS_ORIENT; content vs
    position split by output_regime).
  - tools/xai_study/compare/out/leaderboard.json — per-method aggregate
    F/CI/content/position, triad_S/triad_M, plausibility_proxy, tradition,
    n_games, and the headline all-regime contrast + record count.
  - tools/xai_study/compare/out/position_bootstrap.json — the bootstrapped
    headline gaps and their 95% CIs (all-regime + position regime).
  - docs/groundtruth_data.json — per-game ground truth (origin, verified/moving
    labels, accepted, position_regime, scored).

Method keys in the emitted store are the SITE keys used in manifest.py's
P2_METHODS / the m_<key>.html URLs (e.g. "saliency", "gradxinput"). The
manifest's P2_LEADER map (site key -> leaderboard method name) is imported to
join the two, so the store is addressable by the same key the pages use.

Deterministic and re-runnable:
    python3 docs/gen_site_data.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
COMPARE = os.path.join(REPO, "tools", "xai_study", "compare")

# Import the leaderboard's own loaders so the per-(method,game) faithfulness is
# computed by exactly the same code the aggregate is built from.
sys.path.insert(0, COMPARE)
import leaderboard as L  # noqa: E402

# Import the site's manifest to get the site-key <-> leaderboard-method map and
# the per-method display metadata (name/phase).
sys.path.insert(0, HERE)
import manifest as MAN  # noqa: E402

LEADERBOARD_JSON = os.path.join(COMPARE, "out", "leaderboard.json")
POSITION_BOOTSTRAP_JSON = os.path.join(COMPARE, "out", "position_bootstrap.json")
GROUNDTRUTH_JSON = os.path.join(HERE, "groundtruth_data.json")
OUT = os.path.join(HERE, "site_data.json")


def _mean(vals):
    xs = [v for v in vals
          if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(xs) / len(xs), 4) if xs else None


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def build():
    # ---- leaderboard aggregate rows, keyed by leaderboard method name ----
    lb = json.load(open(LEADERBOARD_JSON))
    lb_rows = {r["method"]: r for r in lb.get("rows", [])}

    # ---- per-(method,game) faithfulness, computed via the leaderboard loaders ----
    # method -> game -> {"content": [..], "position": [..], "all": [..]}
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _phase, _path, rec in L.load_per_game_records():
        mk = L.method_key(rec)
        if mk == "na_audit":
            continue
        game = (rec.get("game") or "").lower()
        if not game or (L.SCORED_GAMES and game not in L.SCORED_GAMES):
            continue
        F, S, M = L.triad_of(rec)
        if F is not None:
            faith = L.clip01(F)
        else:
            faith, _rule = L.orient_faithfulness(rec.get("metric_name"), rec.get("value"))
        if faith is None:
            continue
        regime = L.output_regime(rec)
        per[mk][game]["all"].append(faith)
        if regime in ("content", "position"):
            per[mk][game][regime].append(faith)
        # per-game triad S/M (raw — S can be negative, M in (0,1]); the triad-S/M
        # broadening emits these per record so per-game F/S/M can be shown everywhere.
        if S is not None:
            per[mk][game]["S"].append(S)
        if M is not None:
            per[mk][game]["M"].append(M)

    def per_game_block(lb_method):
        out = {}
        for game, regs in per.get(lb_method, {}).items():
            out[game] = {
                "F": _mean(regs["all"]),
                "F_content": _mean(regs["content"]),
                "F_position": _mean(regs["position"]),
                "S": _mean(regs["S"]) if regs["S"] else None,
                "M": _mean(regs["M"]) if regs["M"] else None,
            }
        return out

    # ---- methods{} — keyed by SITE key (manifest.py P2_METHODS / m_<key>.html) ----
    methods = {}
    site_name = {m["key"]: m["title"] for m in MAN.P2_METHODS}
    site_phase = {m["key"]: m["phase"] for m in MAN.P2_METHODS}
    for site_key, lb_method in MAN.P2_LEADER.items():
        r = lb_rows.get(lb_method)
        if r is None:
            continue
        methods[site_key] = {
            "name": site_name.get(site_key, site_key),
            "leaderboard_method": lb_method,
            "tradition": r.get("tradition"),
            "phase": site_phase.get(site_key),
            "F": r.get("faithfulness"),
            "F_ci": r.get("faithfulness_ci95"),
            "F_content": r.get("faithfulness_content_regime"),
            "F_position": r.get("faithfulness_position_regime"),
            "S": r.get("triad_S"),
            "M": r.get("triad_M"),
            "plaus": r.get("plausibility_proxy"),
            "n_games": r.get("n_games"),
            "n_records": r.get("n_records"),
            "per_game": per_game_block(lb_method),
        }

    # ---- games{} — only the 42 scored games; per-method faithfulness on each ----
    gt = json.load(open(GROUNDTRUTH_JSON))
    # per-game -> per SITE method faithfulness (invert the per{} table)
    game_methods = defaultdict(dict)
    for site_key, lb_method in MAN.P2_LEADER.items():
        for game, vals in per.get(lb_method, {}).items():
            game_methods[game][site_key] = {
                "F": _mean(vals["all"]),
                "F_content": _mean(vals["content"]),
                "F_position": _mean(vals["position"]),
                "S": _mean(vals["S"]) if vals["S"] else None,
                "M": _mean(vals["M"]) if vals["M"] else None,
            }

    games = {}
    for game, d in gt.items():
        if not d.get("scored"):
            continue
        games[game] = {
            "origin": d.get("origin"),
            "n_verified": d.get("n_verified"),
            "n_moving": d.get("n_moving"),
            "accepted": d.get("accepted"),
            "position_regime": d.get("position_regime"),
            "scored": d.get("scored"),
            "per_method": game_methods.get(game, {}),
        }

    # ---- headline — bootstrapped gaps/CIs + family means (all + position) ----
    boot = json.load(open(POSITION_BOOTSTRAP_JSON))
    hc = lb.get("headline_contrast", {})
    all_r = hc.get("all_regimes", {})
    pos_r = hc.get("position_regime", {})
    headline = {
        "all_regime_gap": _round(boot["all_regime"]["family_mean_gap"], 3),
        "all_regime_ci": [_round(x, 3) for x in boot["all_regime"]["ci95"]],
        "position_gap": _round(boot["position"]["family_mean_gap"], 3),
        "position_ci": [_round(x, 3) for x in boot["position"]["ci95"]],
        "causal_all": _round(all_r.get("causal_intervention_faithfulness_mean"), 3),
        "grad_all": _round(all_r.get("gradient_correlational_faithfulness_mean"), 3),
        "causal_pos": _round(pos_r.get("causal_intervention_faithfulness_mean"), 3),
        "grad_pos": _round(pos_r.get("gradient_correlational_faithfulness_mean"), 3),
        # counts used by the headline blurb (family sizes)
        "causal_all_n": all_r.get("causal_intervention_n"),
        "grad_all_n": all_r.get("gradient_correlational_n"),
        "causal_pos_n": pos_r.get("causal_intervention_n"),
        "grad_pos_n": pos_r.get("gradient_correlational_n"),
    }

    scored_games = sorted(games)
    n_records = lb.get("n_per_game_records_aggregated")
    payload = {
        "meta": {
            "scored_games": len(scored_games),
            "n_records": n_records,
            "n_methods": lb.get("n_methods"),
            "generated_by": "docs/gen_site_data.py",
            "source": "leaderboard.json+position_bootstrap.json+§R records",
            "scored_game_list": scored_games,
        },
        "headline": headline,
        "methods": methods,
        "games": games,
    }
    return payload


def main():
    payload = build()
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    m = payload["meta"]
    print("wrote docs/site_data.json")
    print("  scored_games=%s  methods=%s  n_records=%s"
          % (m["scored_games"], len(payload["methods"]), m["n_records"]))
    h = payload["headline"]
    print("  headline all-regime gap=%s ci=%s ; position gap=%s ci=%s"
          % (h["all_regime_gap"], h["all_regime_ci"],
             h["position_gap"], h["position_ci"]))


if __name__ == "__main__":
    main()
