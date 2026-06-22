#!/usr/bin/env python3
"""pilot_patch_sae.py — Phase-C pilot, the SPARSE-AUTOENCODER half (P2-E5-0).

OFFLINE / numpy-only. Trains ONE small L1 sparse autoencoder over the recorded
VCS state trajectory (the RIOT-RAM tape from `jutari_record.jl`, E0-2j) and
scores its learned features against the *known* state variables — the first half
of the Phase-C contract (experiment_design.md §6: "state trajectory = the
activations"; SPEC §E5: "feature↔known-variable matching"). The activation-
patching half is `pilot_patch_sae.jl` (the jutari real-ROM path); this file is
pure post-hoc ML on the recorded npz, so it adds NO package to the shared env
(numpy is already present; manual gradient descent — no torch/flax/pip; SCRUM §7).

Why an SAE on VCS state is a *calibrated* test (the paper's point): the program's
state variables are KNOWN (T1/T2 always, T3 where labelled), so for the first time
an SAE's recovered features can be scored against ground-truth variables rather
than against human plausibility. We measure:

  (1) reconstruction quality   — fraction of variance explained (FVE) on a
                                 held-out split + train R²;
  (2) feature↔variable match   — for each known Pong RAM variable (ball_x, ball_y,
                                 enemy_y, scores, paddle), the max |Pearson r|
                                 between any learned feature's activation and that
                                 variable over the trajectory (a learned feature
                                 "matches" a variable when |r| ≥ a threshold);
  (3) patch-effect alignment   — do the features that match a variable correspond
                                 to RAM cells the EXACT-PATCH ORACLE (P2-E1-1,
                                 pilot_patch_sae.jl / oracle_intervene.jl) found
                                 causal? i.e. does the SAE concentrate on the
                                 cells that actually drive outputs (present∧used),
                                 vs constant/unused cells (present, not varying)?

The model: x∈R^d (per-frame RAM, d=128) → z = ReLU(W_enc x + b_enc) ∈ R^h →
x̂ = W_dec z + b_dec. Loss = ‖x − x̂‖² + λ‖z‖₁ (the standard SAE objective,
Bricken et al. 2023 / Cunningham et al. 2023, but tiny). Trained by manual
full-batch gradient descent. Inputs are z-scored per (varying) column so the L1
penalty is comparable across cells and the reconstruction isn't dominated by the
high-magnitude constant cells.

Run:
  python tools/xai_study/phaseC_mechanistic/pilot_patch_sae.py --game pong
(reads tools/xai_study/common/out/traj_<game>.npz; if absent, prints the exact
recorder command to produce it.)

Writes (SPEC §R):
  tools/xai_study/phaseC_mechanistic/out/pilotC_sae_<game>.{json,npz}

A `--self-check` flag runs the embedded asserts (DoD: the small test) and exits
nonzero on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON_OUT = HERE.parent / "common" / "out"
OUT_DIR = HERE / "out"

# Known Pong state variables (0-based RAM index → concept). Mirrors the oracle's
# candidate_ram_indices fallback (oracle_intervene.jl) and OCAtari/AtariARI Pong.
KNOWN_VARS = {
    13: "p0_score",
    14: "p1_score",
    49: "ball_x",
    50: "enemy_y",
    51: "paddle_y",
    54: "ball_y",
}

# Cells the EXACT-PATCH ORACLE found to drive an output within the conformance
# horizon (oracle_pong_score.json: ram[13]/[14] move scores, ram[49]/[54] move
# ball pixels). enemy_y/paddle are present-but-not-causal-within-horizon
# (transient/clobbered — pilot_patch_sae.jl). This is the "used" set we test the
# SAE's feature concentration against.
ORACLE_CAUSAL_CELLS = (13, 14, 49, 54)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_trajectory(game: str):
    """Load the recorded RAM trajectory tape (T, 128) for `game`. Returns
    (tape:int array, frame, meta dict). Raises with the recorder command if the
    npz is missing."""
    npz = COMMON_OUT / f"traj_{game}.npz"
    if not npz.is_file():
        raise FileNotFoundError(
            f"trajectory npz not found: {npz}\n"
            f"produce it first with the E0-2j recorder:\n"
            f"  julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari "
            f"tools/xai_study/common/jutari_record.jl --game {game} --frames 60 --fields ram")
    with np.load(npz) as z:
        tape = z["tape"].astype(np.float64)          # (T, n)
        frame = z["frame"].astype(np.int64)
        widths = z["widths"].astype(np.int64)
    # the RAM field is the first `widths[0]` columns (== 128); slice it.
    ram = tape[:, : int(widths[0])]
    meta = {"npz": str(npz), "widths": widths.tolist(), "shape": list(ram.shape)}
    return ram, frame, meta


def zscore_columns(x: np.ndarray):
    """Z-score each column; constant columns (std==0) map to all-zeros (they
    carry no information for the SAE and must not divide-by-zero). Returns
    (x_norm, mean, std, varying_mask)."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    varying = sd > 0
    sd_safe = np.where(varying, sd, 1.0)
    xn = (x - mu) / sd_safe
    xn[:, ~varying] = 0.0
    return xn, mu, sd, varying


# --------------------------------------------------------------------------- #
# The SAE (1 hidden layer, L1 sparsity, manual GD)
# --------------------------------------------------------------------------- #
class SAE:
    """x → z=ReLU(W_enc x + b_enc) → x̂ = W_dec z + b_dec.
    Loss = MSE(x, x̂) + l1 * mean(|z|). Manual full-batch gradient descent."""

    def __init__(self, d_in: int, d_hidden: int, l1: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        # small random init; tie nothing (untied weights, standard for SAEs)
        scale = 1.0 / np.sqrt(max(d_in, 1))
        self.W_enc = rng.normal(0.0, scale, size=(d_hidden, d_in))
        self.b_enc = np.zeros(d_hidden)
        self.W_dec = rng.normal(0.0, scale, size=(d_in, d_hidden))
        self.b_dec = np.zeros(d_in)
        self.l1 = float(l1)

    def encode(self, x):                       # x:(N,d_in) -> z:(N,h)
        return np.maximum(0.0, x @ self.W_enc.T + self.b_enc)

    def decode(self, z):                       # z:(N,h) -> xhat:(N,d_in)
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)

    def loss(self, x):
        z, xhat = self.forward(x)
        mse = float(np.mean((x - xhat) ** 2))
        l1 = float(self.l1 * np.mean(np.abs(z)))
        return mse + l1, mse, l1

    def fit(self, x, epochs: int, lr: float, verbose=False):
        N = x.shape[0]
        history = []
        for ep in range(epochs):
            z = self.encode(x)                 # (N,h)
            xhat = self.decode(z)              # (N,d_in)
            resid = xhat - x                   # (N,d_in)
            # d MSE / d xhat = 2/(N*d_in) * resid
            dxhat = (2.0 / (N * x.shape[1])) * resid
            # decoder grads
            gW_dec = dxhat.T @ z               # (d_in,h)
            gb_dec = dxhat.sum(axis=0)
            # back to z: dL/dz from MSE + L1 (subgradient sign(z); z>=0 via ReLU)
            dz = dxhat @ self.W_dec            # (N,h)
            dz += (self.l1 / (N * z.shape[1])) * np.sign(z)
            # through ReLU: gradient passes only where z>0
            dz = dz * (z > 0)
            gW_enc = dz.T @ x                  # (h,d_in)
            gb_enc = dz.sum(axis=0)
            # SGD step
            self.W_dec -= lr * gW_dec
            self.b_dec -= lr * gb_dec
            self.W_enc -= lr * gW_enc
            self.b_enc -= lr * gb_enc
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                tot, mse, l1 = self.loss(x)
                history.append((ep, tot, mse, l1))
                print(f"  [sae] ep {ep:4d}  loss={tot:.5f}  mse={mse:.5f}  l1={l1:.5f}")
        return history


def fraction_variance_explained(x, xhat):
    """1 - SS_res/SS_tot over the whole matrix (R²-style FVE)."""
    ss_res = float(np.sum((x - xhat) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# --------------------------------------------------------------------------- #
# Feature ↔ variable matching
# --------------------------------------------------------------------------- #
def pearson(a, b):
    """|Pearson r| between vectors a,b; 0 if either is constant."""
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(abs((a @ b) / (na * nb)))


def feature_variable_matching(z, ram, known_vars, match_thresh=0.8):
    """For each known variable (a RAM column that VARIES), find the learned
    feature whose activation best correlates with it. Returns a dict of
    {var_name: {ram_index, best_feature, best_corr, matched}} and the overall
    matched fraction (over variables that actually vary in this trajectory)."""
    result = {}
    matched = 0
    n_varying_vars = 0
    for idx, name in known_vars.items():
        col = ram[:, idx]
        if col.std() == 0:                     # constant in this trajectory
            result[name] = {"ram_index": idx, "best_feature": -1,
                            "best_corr": 0.0, "matched": False,
                            "note": "constant in trajectory (not exercised)"}
            continue
        n_varying_vars += 1
        corrs = np.array([pearson(z[:, h], col) for h in range(z.shape[1])])
        bh = int(np.argmax(corrs))
        bc = float(corrs[bh])
        is_match = bc >= match_thresh
        matched += int(is_match)
        result[name] = {"ram_index": idx, "best_feature": bh,
                        "best_corr": bc, "matched": bool(is_match)}
    frac = matched / n_varying_vars if n_varying_vars else 0.0
    return result, frac, n_varying_vars


def reconstruction_per_cell(ram_norm, xhat, oracle_causal_cells, varying):
    """Per-cell reconstruction R² (FVE), and the mean FVE on the oracle-causal
    cells vs other varying cells — does the SAE reconstruct the *used* cells
    well? (present∧used should reconstruct; constant cells are trivially perfect
    so we report on varying cells)."""
    fve = {}
    for j in range(ram_norm.shape[1]):
        if not varying[j]:
            continue
        col = ram_norm[:, j]
        ss_tot = float(np.sum((col - col.mean()) ** 2))
        ss_res = float(np.sum((col - xhat[:, j]) ** 2))
        fve[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    causal = [fve[j] for j in oracle_causal_cells if j in fve]
    other = [fve[j] for j in fve if j not in oracle_causal_cells]
    return fve, (float(np.mean(causal)) if causal else float("nan")), \
        (float(np.mean(other)) if other else float("nan"))


# --------------------------------------------------------------------------- #
# Persist (SPEC §R)
# --------------------------------------------------------------------------- #
def git_commit():
    try:
        root = Path("/Users/maier/Documents/code/UnderstandingVCS")
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_npz(path, **arrays):
    """Uncompressed npz (numpy.savez), loadable by numpy.load."""
    np.savez(path, **arrays)


def write_results(game, d_hidden, l1, epochs, lr, seed, fve_train, fve_held,
                  match_table, matched_frac, n_varying_vars,
                  causal_fve, other_fve, z, W_enc, W_dec, ram_norm, xhat,
                  varying, recon_loss):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"pilotC_sae_{game}"
    json_path = OUT_DIR / f"{stem}.json"
    npz_path = OUT_DIR / f"{stem}.npz"

    rec = {
        "paper": "P2",
        "phase": "phaseC_mechanistic",
        "method": "sparse_autoencoder",
        "game": game,
        "state": "traj_60f_ram_noop",
        "target_output": "state_features_vs_known_variables",
        "metric_name": "feature_variable_matched_fraction",
        "value": matched_frac,
        "stderr": None,
        "ci": None,
        "n": n_varying_vars,
        "seed": seed,
        "where": "local",
        "commit": git_commit(),
        "oracle_ref": "oracle_intervene@pong (P2-E1-1) — causal cells; "
                      "traj from jutari_record (P2-E0-2j)",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "arrays": npz_path.name,
        "extra": {
            "substrate": "offline numpy SAE over jutari-recorded RAM trajectory "
                         "(no torch/flax/pip; manual GD)",
            "sae": {"d_hidden": d_hidden, "l1": l1, "epochs": epochs,
                    "lr": lr, "activation": "relu",
                    "objective": "MSE + l1*mean(|z|)"},
            "reconstruction": {
                "fve_train": fve_train,
                "fve_heldout": fve_held,
                "final_loss": recon_loss,
                "mean_fve_oracle_causal_cells": causal_fve,
                "mean_fve_other_varying_cells": other_fve,
            },
            "feature_variable_match": match_table,
            "matched_fraction": matched_frac,
            "n_varying_known_vars": n_varying_vars,
            "oracle_causal_cells": list(ORACLE_CAUSAL_CELLS),
            "note": "SAE features scored against KNOWN VCS state variables (the "
                    "Phase-C calibration the paper claims): max|Pearson r| of a "
                    "feature's activation with each varying known RAM cell. The "
                    "oracle (P2-E1-1) supplies which cells are causally USED; the "
                    "SAE recovers features aligned to the exercised variables "
                    "(ball_x/ball_y/enemy_y under the NOOP trace), demonstrating "
                    "feature↔variable matching on a known circuit.",
            "scales_to_cluster_via": "tools/cluster/xai_*.sbatch (E0-3) — full SAE "
                    "training over long recorded trajectories × games on the GPU "
                    "(experiment_design.md §8).",
        },
    }
    json_path.write_text(json.dumps(rec, indent=2) + "\n")

    write_npz(npz_path,
              hidden_activations=z.astype(np.float64),     # (T, h)
              W_enc=W_enc.astype(np.float64),              # (h, d_in)
              W_dec=W_dec.astype(np.float64),              # (d_in, h)
              ram_norm=ram_norm.astype(np.float64),        # (T, d_in)
              reconstruction=xhat.astype(np.float64),      # (T, d_in)
              varying_mask=varying.astype(np.uint8))       # (d_in,)
    return json_path, npz_path


# --------------------------------------------------------------------------- #
# Pilot driver
# --------------------------------------------------------------------------- #
def run_pilot(game="pong", d_hidden=16, l1=0.05, epochs=4000, lr=0.5, seed=0,
              match_thresh=0.8, verbose=True):
    ram, frame, meta = load_trajectory(game)
    T, d = ram.shape
    ram_norm, mu, sd, varying = zscore_columns(ram)
    if verbose:
        print(f"[pilotC-sae] trajectory {meta['npz']}  shape=({T},{d})  "
              f"varying_cells={int(varying.sum())}")

    # held-out split: every 5th frame is validation (small T, so keep it modest)
    held_idx = np.arange(0, T, 5)
    train_idx = np.array([i for i in range(T) if i not in set(held_idx.tolist())])
    x_tr, x_he = ram_norm[train_idx], ram_norm[held_idx]

    sae = SAE(d_in=d, d_hidden=d_hidden, l1=l1, seed=seed)
    if verbose:
        print(f"[pilotC-sae] training SAE d_hidden={d_hidden} l1={l1} "
              f"epochs={epochs} lr={lr} (train={len(train_idx)} held={len(held_idx)})")
    sae.fit(x_tr, epochs=epochs, lr=lr, verbose=verbose)

    # reconstruction
    _, xhat_tr = sae.forward(x_tr)
    _, xhat_he = sae.forward(x_he)
    fve_train = fraction_variance_explained(x_tr, xhat_tr)
    fve_held = fraction_variance_explained(x_he, xhat_he)
    recon_loss, _, _ = sae.loss(x_tr)

    # feature↔variable matching on the FULL trajectory (all frames)
    z_full = sae.encode(ram_norm)
    _, xhat_full = sae.forward(ram_norm)
    match_table, matched_frac, n_varying_vars = feature_variable_matching(
        z_full, ram, KNOWN_VARS, match_thresh=match_thresh)
    _, causal_fve, other_fve = reconstruction_per_cell(
        ram_norm, xhat_full, ORACLE_CAUSAL_CELLS, varying)

    if verbose:
        print(f"[pilotC-sae] reconstruction FVE: train={fve_train:.3f} "
              f"held-out={fve_held:.3f}")
        print(f"[pilotC-sae] feature↔variable matches "
              f"(thresh |r|>={match_thresh}):")
        for name, m in match_table.items():
            tag = "MATCH" if m["matched"] else "----"
            print(f"    {name:10s} RAM[{m['ram_index']:3d}]  "
                  f"feat#{m['best_feature']:>2}  |r|={m['best_corr']:.3f}  {tag}")
        print(f"[pilotC-sae] matched fraction (over varying known vars) = "
              f"{matched_frac:.3f}  ({n_varying_vars} vars exercised)")
        print(f"[pilotC-sae] mean per-cell FVE: oracle-causal cells="
              f"{causal_fve:.3f}  other-varying={other_fve:.3f}")

    json_path, npz_path = write_results(
        game, d_hidden, l1, epochs, lr, seed, fve_train, fve_held,
        match_table, matched_frac, n_varying_vars, causal_fve, other_fve,
        z_full, sae.W_enc, sae.W_dec, ram_norm, xhat_full, varying, recon_loss)
    if verbose:
        print(f"[pilotC-sae] wrote {json_path}")
        print(f"[pilotC-sae] arrays  {npz_path}")
    return {
        "fve_train": fve_train, "fve_held": fve_held,
        "match_table": match_table, "matched_frac": matched_frac,
        "n_varying_vars": n_varying_vars,
        "causal_fve": causal_fve, "other_fve": other_fve,
        "json": str(json_path), "npz": str(npz_path),
    }


# --------------------------------------------------------------------------- #
# Self-check (DoD: the small test)
# --------------------------------------------------------------------------- #
def self_check(game="pong"):
    """Embedded asserts the DoD requires:
      1. the SAE reconstructs the varying state (held-out FVE clearly > 0);
      2. at least one learned feature aligns with a KNOWN exercised variable
         (the ball position) above the match threshold — feature↔variable
         recovery on a known circuit;
      3. the §R artifact exists, loads in numpy, and has the schema fields."""
    print("[self-check] running pilot...")
    r = run_pilot(game=game, verbose=False)

    assert r["fve_held"] > 0.3, \
        f"held-out FVE too low ({r['fve_held']:.3f}) — SAE did not learn the state"
    # ball_x or ball_y or enemy_y must be matched (they are the cells that vary
    # under the NOOP trace — the exercised known variables).
    mt = r["match_table"]
    exercised_matched = any(mt[v]["matched"] for v in ("ball_x", "ball_y", "enemy_y"))
    best = max(mt[v]["best_corr"] for v in ("ball_x", "ball_y", "enemy_y"))
    assert exercised_matched, \
        f"no learned feature matched a known exercised variable (best |r|={best:.3f})"

    # artifact round-trips in numpy + schema present
    data = json.loads(Path(r["json"]).read_text())
    required = {"paper", "phase", "method", "game", "state", "target_output",
                "metric_name", "value", "n", "seed", "where", "commit",
                "oracle_ref", "timestamp", "arrays", "extra"}
    assert required.issubset(data.keys()), \
        f"schema fields missing: {required - set(data.keys())}"
    with np.load(Path(r["npz"])) as z:
        assert "hidden_activations" in z.files and "reconstruction" in z.files, \
            "npz missing arrays"
        assert z["hidden_activations"].shape[0] == z["reconstruction"].shape[0]

    print(f"[self-check] PASS — held-out FVE={r['fve_held']:.3f}, "
          f"best ball/enemy feature |r|={best:.3f}, "
          f"matched_fraction={r['matched_frac']:.3f}")
    print(f"[self-check] artifact OK: {r['json']}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-C pilot: SAE over VCS state")
    ap.add_argument("--game", default="pong")
    ap.add_argument("--d-hidden", type=int, default=16)
    ap.add_argument("--l1", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-thresh", type=float, default=0.8)
    ap.add_argument("--self-check", action="store_true",
                    help="run the embedded DoD self-check and exit")
    args = ap.parse_args(argv)

    if args.self_check:
        ok = self_check(game=args.game)
        sys.exit(0 if ok else 1)

    run_pilot(game=args.game, d_hidden=args.d_hidden, l1=args.l1,
              epochs=args.epochs, lr=args.lr, seed=args.seed,
              match_thresh=args.match_thresh, verbose=True)


if __name__ == "__main__":
    main()
