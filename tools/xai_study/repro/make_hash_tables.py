#!/usr/bin/env python3
"""make_hash_tables.py — regenerate the P2 reproducibility hash tables.

Review point: improvement_instructions P1#20 ("Add ROM and action-stream
reproducibility") + P1#21 ("artifact traceability") + P6 §54–55 (data files +
validation scripts: ROM hash table not ROMs, action streams + seeds, checksum
manifest). Companion doc: ../../../xai_paper/xai_2_interpretability/REPRODUCIBILITY.md.

What it does (and *only* this — it ADDS reproducibility metadata, it never
re-runs or alters any committed experimental result):

  1. Resolves each of the 6 core ROMs **in place** (gitignored; never copied,
     never committed — SCRUM §7) via the same `loader.resolve_rom` the oracle
     uses, computes its **real SHA-256 + byte size**, and writes
     `rom_hash_table.csv` with the AutoROM acquisition name + validation horizon.
     ROM *bytes* are never written anywhere — only the digest.
  2. Derives the **deterministic action streams** used by the experiments from
     their documented `state` encoding (boot = 60 NOOP + 4 RESET; the replay
     trace is all-NOOP — `actions = fill(0, total)` in
     ground_truth/oracle_intervene.jl:265 — with interventions injected at the
     target frame), hashes each stream, and writes `action_stream_hashes.csv`.
  3. Prints a checksum **manifest** (the SHA-256 of both CSVs and of this
     script) so a reviewer can confirm the tables were regenerated unchanged.

Run (no extra deps; numpy via the jaxtari venv, but plain python3 also works):

    XAI_PRIMARY_REPO=<repo> python3 tools/xai_study/repro/make_hash_tables.py
    # or, from a worktree, the env var lets the gitignored ROMs resolve in the
    # primary checkout. With --verify, it re-reads the CSVs and re-hashes the
    # ROMs, asserting the committed digests still match the bytes on disk.

This is a pure read of the substrate; it has no side effect on any §R record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

# tools/xai_study/repro/make_hash_tables.py -> repo root is parents[3].
REPO = Path(__file__).resolve().parents[3]
COMMON = REPO / "tools" / "xai_study" / "common"
OUT_DIR = Path(__file__).resolve().parent
ROM_CSV = OUT_DIR / "rom_hash_table.csv"
ACTION_CSV = OUT_DIR / "action_stream_hashes.csv"

# Make `import loader` resolve the in-repo ROM loader (resolve_rom).
sys.path.insert(0, str(COMMON))

# The 6 headline core games (SPEC §G; tools/xai_study/common/game_set.json).
CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# AutoROM acquisition name per core game (ALE/Gymnasium licensed ROM set).
# `pip install autorom && AutoROM --accept-license` installs these; the user
# then verifies the bytes against the SHA-256 below. We never redistribute ROMs.
AUTOROM_NAME = {
    "pong": "pong",
    "breakout": "breakout",
    "space_invaders": "space_invaders",
    "seaquest": "seaquest",
    "ms_pacman": "ms_pacman",
    "qbert": "qbert",
}

# Paper-1 bit-exact validation horizon (jutari ≡ xitari 64/64): the conformance
# window that all P2 experiment states stay inside (~30-frame RAM / 60-frame
# screen). Recorded per game so the table is self-contained.
VALIDATION_HORIZON = "RAM ~30 frames / screen ~60 frames (Paper-1 64/64 bit-exact window)"

# ---------------------------------------------------------------------------
# Action streams. Every committed §R record is produced under a *deterministic*
# action trace: the xitari-parity boot (60 NOOP + 4 RESET) followed by an
# all-NOOP replay of length `total` (the experiment may inject an intervention
# AT the target frame — that is part of scoring, not of the baseline stream).
# `seed = 0` everywhere (the substrate is deterministic; seeded methods —
# random/RISE/SmoothGrad/Expected-Gradients — reproduce exactly). We hash the
# canonical baseline stream per (game, state-encoding); the hash is the SHA-256
# of the explicit byte sequence "<boot>;<replay>" so it is reproducible from
# this script alone, independent of the emulator.
# ---------------------------------------------------------------------------

BOOT_NOOP = 60
BOOT_RESET = 4
# jaxtari/jutari action ids: 0 = NOOP (PLAYER_A_NOOP). RESET during boot is the
# console RESET switch, encoded by the boot harness, not an agent action; for
# the action-stream digest we represent the boot prefix explicitly.
NOOP = 0
RESET_TOKEN = "RESET"

# The state encodings used across the phases (from the committed records'
# `state` field). `fW+H` = boot, then W NOOP frames to the checkpoint, then an
# H-frame scoring window. We document the canonical streams that back the
# headline numbers; the per-record `state` field is the authoritative pointer.
# (target_frame, horizon, note) — the action stream is NOOP^(W) ; NOOP^(H).
STATE_STREAMS = [
    ("f120+30", 120, 30, "ground_truth oracle + Phase B/C + benchmark headline checkpoint"),
    ("f30+30", 30, 30, "Phase A / Phase C early-window checkpoint"),
    ("f60+30", 60, 30, "Phase A / Phase C mid-window checkpoint"),
    ("f330+30", 330, 30, "Ms. Pac-Man late-window checkpoint (Phase A/C)"),
    ("f300+30", 300, 30, "Pong late-window checkpoint (Phase B)"),
    ("f0+256", 0, 256, "Phase A whole-state recording window (A8)"),
    ("f0+400", 0, 400, "Ms. Pac-Man Phase A whole-state recording window"),
    ("traj_60f_ram_noop", 0, 60, "Phase C 60-frame NOOP RAM trajectory"),
]


def _build_stream(target_frame: int, horizon: int) -> list:
    """The explicit deterministic action token stream for a (target,horizon)."""
    boot = [RESET_TOKEN if i >= BOOT_NOOP else NOOP
            for i in range(BOOT_NOOP + BOOT_RESET)]
    replay = [NOOP] * (target_frame + horizon)
    return boot + replay


def _stream_sha256(stream: list) -> str:
    payload = ",".join(str(x) for x in stream).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def sha256_of_text_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
def build_rom_table() -> list[dict]:
    import loader  # tools/xai_study/common/loader.py
    rows = []
    for game in CORE_GAMES:
        path = loader.resolve_rom(game)  # resolves in place; reads bytes only
        digest, size = sha256_of_file(path)
        # Report the path RELATIVE to the repo root so the CSV is location-stable
        # and never leaks an absolute home dir. ROM bytes are NOT written.
        try:
            rel = path.resolve().relative_to(REPO.resolve())
            rel_str = str(rel)
        except ValueError:
            # ROM resolved in the primary checkout (worktree case) — record the
            # repo-relative subpath without the primary prefix.
            rel_str = "/".join(path.parts[-2:])
        rows.append({
            "game": game,
            "autorom_name": AUTOROM_NAME[game],
            "sha256": digest,
            "size_bytes": size,
            "resolved_path": rel_str,
            "retrieval": ("AutoROM (pip install autorom && AutoROM "
                          "--accept-license); verify SHA-256 below"),
            "validation_horizon": VALIDATION_HORIZON,
        })
    return rows


def build_action_table() -> list[dict]:
    rows = []
    for state, tf, hz, note in STATE_STREAMS:
        stream = _build_stream(tf, hz)
        rows.append({
            "experiment_state": state,
            "boot": f"{BOOT_NOOP} NOOP + {BOOT_RESET} RESET (xitari/ALE parity)",
            "replay_action": "NOOP (action id 0)",
            "target_frame": tf,
            "horizon_frames": hz,
            "total_frames": tf + hz,
            "n_action_tokens": len(stream),
            "seed": 0,
            "action_stream_sha256": _stream_sha256(stream),
            "output_ids": note,
        })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def verify() -> int:
    """Re-read the committed CSVs; re-hash the ROMs; assert digests still match."""
    import loader
    if not ROM_CSV.is_file():
        print(f"[verify] FAIL: {ROM_CSV} missing", file=sys.stderr)
        return 1
    ok = True
    with open(ROM_CSV) as fh:
        for row in csv.DictReader(fh):
            game = row["game"]
            path = loader.resolve_rom(game)
            digest, size = sha256_of_file(path)
            match = (digest == row["sha256"] and str(size) == row["size_bytes"])
            print(f"[verify] {game:16s} {'OK' if match else 'MISMATCH'} "
                  f"sha256={digest[:12]}… size={size}")
            ok = ok and match
    print("[verify] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-hash the ROMs and assert the committed CSV digests match")
    args = ap.parse_args()

    if args.verify:
        return verify()

    rom_rows = build_rom_table()
    action_rows = build_action_table()

    write_csv(ROM_CSV, rom_rows,
              ["game", "autorom_name", "sha256", "size_bytes",
               "resolved_path", "retrieval", "validation_horizon"])
    write_csv(ACTION_CSV, action_rows,
              ["experiment_state", "boot", "replay_action", "target_frame",
               "horizon_frames", "total_frames", "n_action_tokens", "seed",
               "action_stream_sha256", "output_ids"])

    # --- checksum manifest --------------------------------------------------
    print("=" * 72)
    print("P2 reproducibility hash tables — checksum manifest")
    print("=" * 72)
    print(f"ROM table     : {ROM_CSV.name}  ({len(rom_rows)} core ROMs)")
    for r in rom_rows:
        print(f"  {r['game']:16s} sha256={r['sha256']}  {r['size_bytes']:>5d} B  "
              f"AutoROM={r['autorom_name']}")
    print(f"action streams: {ACTION_CSV.name}  ({len(action_rows)} deterministic streams)")
    for r in action_rows:
        print(f"  {r['experiment_state']:20s} sha256={r['action_stream_sha256']}  "
              f"seed={r['seed']} total={r['total_frames']}f")
    print("-" * 72)
    print("artifact digests (sha256 of the regenerated files):")
    print(f"  {ROM_CSV.name:28s} {sha256_of_text_file(ROM_CSV)}")
    print(f"  {ACTION_CSV.name:28s} {sha256_of_text_file(ACTION_CSV)}")
    print(f"  {Path(__file__).name:28s} {sha256_of_text_file(Path(__file__))}")
    print("-" * 72)
    print("NOTE: ROMs are gitignored and never committed (SCRUM §7). This script "
          "writes digests only — no ROM bytes. Re-run with --verify to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
