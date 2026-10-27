#!/usr/bin/env python3
"""Paper-2 Results B: true end-to-end gradient vs the single-read gradient, per scored game.

For every scored game (common/game_sets.jl::XAI_SCORED, read via leaderboard's loader)
and both outputs (content, position) this pairs

  phaseB_attribution/out/e2e_saliency_<g>_<out>.json    gradient of the output at f*+15
                                                        back through every CPU instruction
                                                        of the horizon (e2e_gradient.jl)
  phaseB_attribution/out/saliency_<g>_<out>.json        vanilla saliency: the single read
                                                        at the intervention frame

and reports the e2e gradient's support (‖g‖₁, nnz), whether it is the one-hot e_idx,
the F leg of both (Pearson with the oracle, as in the records' triad), and the largest
per-cause attribution difference. e2e_gradxinput's F is listed alongside.

  python3 tools/xai_study/compare/e2e_vs_single.py   -> compare/out/e2e_vs_single.{json,md}
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leaderboard as L  # noqa: E402

PB = os.path.join(HERE, "..", "phaseB_attribution", "out")
OUT = os.path.join(HERE, "out")
TOL = 1e-9


def load(name):
    p = os.path.join(PB, name)
    return json.load(open(p)) if os.path.exists(p) else None


def F(rec):
    return None if rec is None else rec["extra"]["triad"]["F"]


def attr(rec, key):
    return rec["extra"].get(key) or {}


def row(game, out):
    e = load(f"e2e_saliency_{game}_{out}.json")
    s = load(f"saliency_{game}_{out}.json")
    gx = load(f"e2e_gradxinput_{game}_{out}.json")
    if e is None:
        return {"game": game, "output": out, "status": "missing_e2e"}
    ex = e["extra"]
    r = {"game": game, "output": out, "status": "ok",
         "target_e2e": e["target_output"], "target_single": None if s is None else s["target_output"],
         "grad_l1": ex["grad_l1"], "grad_nnz": ex["grad_nnz"],
         "n_instructions": ex.get("n_instructions_differentiated"),
         "backward_seconds": ex.get("backward_seconds"),
         "F_e2e_saliency": F(e), "F_single_saliency": F(s), "F_e2e_gradxinput": F(gx)}
    r["one_hot"] = ex["grad_nnz"] == 1 and abs(ex["grad_l1"] - 1) < TOL
    r["zero_gradient"] = ex["grad_l1"] == 0
    if s is not None:
        a, b = attr(e, "attr_per_cause"), attr(s, "saliency_attr_per_cause")
        keys = sorted(set(a) | set(b))
        r["max_abs_attr_diff"] = max((abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys),
                                     default=0.0)
        fe, fs = r["F_e2e_saliency"], r["F_single_saliency"]
        r["F_equal"] = (fe is not None and fs is not None and
                        (abs(fe - fs) < 1e-9 or (math.isnan(fe) and math.isnan(fs))))
        # the e2e map vs the single read: identical | the gradient vanished | a one-hot on a
        # different byte (the output byte was copied from another during the horizon) |
        # extra support (e.g. a velocity byte) that leaves the per-cause map unchanged | other
        r["category"] = ("identical" if r["max_abs_attr_diff"] < TOL and ex["grad_nnz"] <= 1 else
                         "zero" if r["zero_gradient"] else
                         "moved_one_hot" if r["one_hot"] else
                         "extra_support_same_map" if r["max_abs_attr_diff"] < TOL else "other")
    return r


def main():
    games = sorted(L._load_scored_games())
    rows = [row(g, o) for g in games for o in ("content", "position")]
    ok = [r for r in rows if r["status"] == "ok"]
    cnt = [r for r in ok if r["output"] == "content"]
    pos = [r for r in ok if r["output"] == "position"]
    clip = lambda x: None if x is None else L.clip01(x)
    mean = lambda xs: None if not xs else sum(xs) / len(xs)
    summ = {
        "n_scored_games": len(games),
        "n_games_with_e2e": len({r["game"] for r in ok}),
        "missing": sorted({r["game"] for r in rows if r["status"] != "ok"}),
        "content_n": len(cnt),
        "categories": {o: {c: sorted(r["game"] for r in rs if r.get("category") == c)
                           for c in sorted({r.get("category") for r in rs})}
                       for o, rs in (("content", cnt), ("position", pos))},
        "content_F_equal_single": sum(bool(r.get("F_equal")) for r in cnt),
        "position_n": len(pos),
        "position_F_equal_single": sum(bool(r.get("F_equal")) for r in pos),
        "not_equal": [(r["game"], r["output"], r["F_single_saliency"], r["F_e2e_saliency"])
                      for r in ok if not r.get("F_equal")],
    }
    for name, rs in (("content", cnt), ("position", pos)):
        for k in ("F_e2e_saliency", "F_single_saliency", "F_e2e_gradxinput"):
            v = [clip(r[k]) for r in rs if r[k] is not None and not math.isnan(r[k])]
            summ[f"mean_clipped_{k}_{name}"] = mean(v)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(OUT, "e2e_vs_single.json"), "w"), indent=1)
    with open(os.path.join(OUT, "e2e_vs_single.md"), "w") as f:
        f.write("# End-to-end vs single-read gradient (Paper 2, Results B)\n\n"
                "Generated by `tools/xai_study/compare/e2e_vs_single.py`. F = Pearson with the "
                "oracle (the records' triad F, unclipped here).\n\n")
        f.write("```\n" + json.dumps(summ, indent=1) + "\n```\n\n")
        f.write("| game | output | ‖g‖₁ | nnz | F single | F e2e | F e2e G×I | max Δattr | category |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        fmt = lambda x: "—" if x is None else f"{x:.3f}"
        for r in rows:
            if r["status"] != "ok":
                f.write(f"| {r['game']} | {r['output']} | {r['status']} |||||||\n")
                continue
            f.write(f"| {r['game']} | {r['output']} | {r['grad_l1']:.4g} | {r['grad_nnz']} | "
                    f"{fmt(r['F_single_saliency'])} | {fmt(r['F_e2e_saliency'])} | "
                    f"{fmt(r['F_e2e_gradxinput'])} | {r.get('max_abs_attr_diff', float('nan')):.3g} | {r.get('category', '—')} |\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
