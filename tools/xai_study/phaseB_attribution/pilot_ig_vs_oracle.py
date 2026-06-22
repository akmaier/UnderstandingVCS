#!/usr/bin/env python3
"""pilot_ig_vs_oracle.py — Phase-B pilot (P2-E4-0), the JULIA-path dispatcher.

The Phase-B pilot is implemented in **Julia** (`pilot_ig_vs_oracle.jl`), on the
verified jutari foundation — the same substrate pivot the E1 oracle made
(`ground_truth/oracle_intervene.py` → `.jl`): jutari is the fast, bit-exact
real-ROM path (Paper-1 64/64), while jaxtari eager HARD stepping is ~205× slower
(SCRUM §7). The Integrated-Gradients attribution uses Zygote over the
differentiable substrate, and the deletion/insertion curves re-run the TRUE VCS
via the bit-exact intervention oracle (P2-E1-1).

This thin shim preserves the DoD command path and forwards every argument to the
Julia script via the warm shared `~/.julia` depot:

    python tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.py \
        --game pong --output ball_pixel

is equivalent to (and runs):

    julia --project=<repo>/jutari \
        tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl \
        --game pong --output ball_pixel

Outputs (SPEC §R; file_scope pilotB_*):
    tools/xai_study/phaseB_attribution/out/pilotB_faithfulness_ig_pong_<output>.{json,npz}

Default output is the FAITHFUL headline case (`p0_score` at a live-score frame,
where IG concentrates on the true causal score byte); `--output ball_pixel` is
the POSITION/INDEX contrast where IG vanishes (the §1 'plausible ≠ faithful'
result).
"""
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root that contains this worktree (…/tools/xai_study/phaseB_attribution)
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
# jutari project: prefer this checkout's; fall back to the primary (warm depot)
_PRIMARY = os.environ.get("XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")


def _jutari_project():
    # PREFER the PRIMARY checkout's jutari project: its global `~/.julia` depot is
    # warm + instantiated (SCRUM §7), so a worktree pays no recompile and no
    # `Pkg.instantiate()`. The .jl scripts are loaded by absolute path, so the
    # project choice only governs the package environment.
    for base in (_PRIMARY, _REPO):
        p = os.path.join(base, "jutari")
        if os.path.isdir(p):
            return p
    return os.path.join(_PRIMARY, "jutari")


def main(argv):
    jl = os.path.join(_HERE, "pilot_ig_vs_oracle.jl")
    cmd = ["julia", f"--project={_jutari_project()}", jl, *argv]
    print("[pilotB.py] dispatching to Julia:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
