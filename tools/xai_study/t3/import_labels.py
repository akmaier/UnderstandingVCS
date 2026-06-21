#!/usr/bin/env python3
"""P2-E2-1 — Import OCAtari / AtariARI candidate RAM->concept labels (T3).

Produces *candidate* (correlational) T3 labels for the P2 core game set
(Pong, Breakout, Space Invaders, Seaquest, Ms. Pac-Man, Q*Bert) from two public
sources of Atari-2600 RAM annotations:

  * **AtariARI** (mila-iqia ``atari-representation-learning``, 22 games):
    the ``ram_annotations`` (``atari_dict``) mapping concept-name -> RAM index.
    These are the *raw* RAM addresses; AtariARI does **not** offset-correct the
    (x, y) values to the rendered framebuffer position (experiment_design.md §2).
    The dict is transcribed verbatim below (no ROMs / no install needed).

  * **OCAtari** (``ocatari`` >= 2.x, 40+ games): the per-game RAM-mode object
    extractors in ``ocatari/ram/<game>.py``. These *do* apply render-time
    offsets (e.g. Pong ball_x = ram[49] - 49). We recover the
    ``ram_state[i] (+/- offset)`` -> concept triples by AST-parsing the
    installed ocatari source (the top-level ``import ocatari`` needs ALE+ROMs,
    so we parse the pure-Python ram modules directly instead of importing them).

Output: one self-describing record per game at
``tools/xai_study/t3/out/candidates_<game>.json`` following SPEC §R, with
per-candidate provenance (``source``, ``raw`` vs ``offset_applied``). These are
**candidates only** — correlational until verified by intervention (P2-E2-2).

Run:
    PY=/Users/maier/Documents/code/UnderstandingVCS/jaxtari/.venv/bin/python
    PYTHONPATH=<worktree> $PY tools/xai_study/t3/import_labels.py \
        --ocatari-src "$($PY -c 'import ocatari,os;print(os.path.dirname(ocatari.__file__))' 2>/dev/null \
                       || echo /path/to/site-packages/ocatari)"

If ocatari is not installed, the script still emits AtariARI-only candidates and
records the OCAtari source as unavailable in the record (no fabrication).
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------- #
# Constants / provenance
# --------------------------------------------------------------------------- #

PAPER = "P2"
PHASE = "E2-T3"
METHOD = "import_candidate_labels"
SPEC_RAM_BYTES = 128  # Atari-2600 console RAM (RIOT $80-$FF), 0..127

# core game set (SPEC §G). key = our canonical id used in the output filename.
CORE_GAMES = [
    "pong",
    "breakout",
    "space_invaders",
    "seaquest",
    "ms_pacman",
    "qbert",
]

# id -> (AtariARI atari_dict key, ocatari ram module basename)
GAME_SRC_NAMES = {
    "pong": ("pong", "pong"),
    "breakout": ("breakout", "breakout"),
    "space_invaders": ("space_invaders", "spaceinvaders"),
    "seaquest": ("seaquest", "seaquest"),
    "ms_pacman": ("ms_pacman", "mspacman"),
    "qbert": ("qbert", "qbert"),
}

ATARIARI_REF = (
    "mila-iqia/atari-representation-learning "
    "(AtariARI; Anand et al. NeurIPS 2019), "
    "atariari/benchmark/ram_annotations.py :: atari_dict"
)
OCATARI_REF = "OCAtari (Delfosse et al. 2024), ocatari/ram/<game>.py RAM-mode extractors"


# --------------------------------------------------------------------------- #
# (1) AtariARI ram_annotations — transcribed verbatim (raw addresses, no offset)
#     Source: atariari/benchmark/ram_annotations.py  (public, MIT).
#     Only the 6 core games are kept here; values are RAM byte indices 0..127.
# --------------------------------------------------------------------------- #

ATARIARI_DICT: dict[str, dict[str, int]] = {
    "pong": {
        "player_y": 51,
        "player_x": 46,
        "enemy_y": 50,
        "enemy_x": 45,
        "ball_x": 49,
        "ball_y": 54,
        "enemy_score": 13,
        "player_score": 14,
    },
    "breakout": {
        "ball_x": 99,
        "ball_y": 101,
        "player_x": 72,
        "blocks_hit_count": 77,
        "block_bit_map": 30,  # see breakout bitmaps
        "score": 84,  # 5 for each hit
    },
    "space_invaders": {
        "invaders_left_count": 17,
        "player_score": 104,
        "num_lives": 73,
        "player_x": 28,
        "enemies_x": 26,
        "missiles_y": 9,
        "enemies_y": 24,
    },
    "seaquest": {
        "enemy_obstacle_x": 30,
        "player_x": 70,
        "player_y": 97,
        "diver_or_enemy_missile_x": 71,
        "player_direction": 86,
        "player_missile_direction": 87,
        "oxygen_meter_value": 102,
        "player_missile_x": 103,
        "score": 57,
        "num_lives": 59,
        "divers_collected_count": 62,
    },
    "ms_pacman": {
        "enemy_sue_x": 6,
        "enemy_inky_x": 7,
        "enemy_pinky_x": 8,
        "enemy_blinky_x": 9,
        "enemy_sue_y": 12,
        "enemy_inky_y": 13,
        "enemy_pinky_y": 14,
        "enemy_blinky_y": 15,
        "player_x": 10,
        "player_y": 16,
        "fruit_x": 11,
        "fruit_y": 17,
        "ghosts_count": 19,
        "player_direction": 56,
        "dots_eaten_count": 119,
        "player_score": 120,
        "num_lives": 123,
    },
    "qbert": {
        "player_x": 43,
        "player_y": 67,
        "player_column": 35,
        "red_enemy_column": 69,
        "green_enemy_column": 105,
        "score": 89,  # internal score = score * 100; see scoring
        "tile_color_1": 21,
        "tile_color_2": 52,
        "tile_color_3": 54,
        "tile_color_4": 83,
        "tile_color_5": 85,
        "tile_color_6": 86,
    },
}


# --------------------------------------------------------------------------- #
# (2) OCAtari extractor harvesting via AST (no import; offsets recovered)
# --------------------------------------------------------------------------- #

@dataclass
class OcatariRef:
    """One ``ram_state[index]`` reference recovered from an ocatari extractor."""

    ram_index: int
    offset: Optional[int]          # additive constant applied to the byte, if any
    concept: Optional[str]         # best-effort concept label (assignment target / info key)
    func: str                      # extractor function it came from
    is_slice: bool = False         # ram_state[a:b] (multi-byte concept)
    slice_end: Optional[int] = None


def _const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_int(node.operand)
        return -inner if inner is not None else None
    return None


def _ram_subscript(node: ast.AST):
    """If node is ``ram_state[...]`` return (index_or_None, is_slice, slice_end)."""
    if not (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ram_state"):
        return None
    sl = node.slice
    if isinstance(sl, ast.Slice):
        lo = _const_int(sl.lower) if sl.lower is not None else None
        hi = _const_int(sl.upper) if sl.upper is not None else None
        return (lo, True, hi)
    idx = _const_int(sl)
    return (idx, False, None)


def _offset_of(binop: ast.BinOp) -> Optional[int]:
    """For ``ram_state[i] +/- C`` return the signed C (the render-time offset)."""
    if isinstance(binop.op, (ast.Add, ast.Sub)):
        c = _const_int(binop.right)
        if c is not None:
            return c if isinstance(binop.op, ast.Add) else -c
    return None


def _label_from_target(target: ast.AST) -> Optional[str]:
    """Best-effort concept label from an assignment LHS like ``info["ball_x"]``,
    ``player.xy``, ``object_info["player_x"]``."""
    if isinstance(target, ast.Subscript):
        sl = target.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            base = ""
            if isinstance(target.value, ast.Name):
                base = ""  # info[...] / object_info[...] -> just the key
            return sl.value
    if isinstance(target, ast.Attribute):
        base = target.value.id if isinstance(target.value, ast.Name) else ""
        return f"{base}.{target.attr}" if base else target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


class _OcatariVisitor(ast.NodeVisitor):
    """Walk an ocatari ram module, collecting ram_state[...] refs with the
    nearest enclosing assignment target as the concept label and any additive
    offset applied at the use site."""

    def __init__(self) -> None:
        self.refs: list[OcatariRef] = []
        self._func = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        prev = self._func
        self._func = node.name
        self.generic_visit(node)
        self._func = prev

    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        label = None
        for t in node.targets:
            label = _label_from_target(t) or label
        self._scan_value(node.value, label)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):  # noqa: N802
        label = _label_from_target(node.target)
        if node.value is not None:
            self._scan_value(node.value, label)
        self.generic_visit(node)

    def _scan_value(self, value: ast.AST, label: Optional[str]) -> None:
        """Find ram_state[...] inside an RHS expression, with offsets."""
        # bare subscript ram_state[i] or ram_state[a:b]
        sub = _ram_subscript(value)
        if sub is not None:
            idx, is_slice, end = sub
            if idx is not None:
                self.refs.append(OcatariRef(idx, None, label, self._func, is_slice, end))
            return
        # ram_state[i] +/- C
        if isinstance(value, ast.BinOp):
            sub = _ram_subscript(value.left)
            off = _offset_of(value)
            if sub is not None and not sub[1] and sub[0] is not None:
                self.refs.append(OcatariRef(sub[0], off, label, self._func))
                return
        # containers: tuple / list / dict / call args -> recurse for nested refs
        for child in ast.iter_child_nodes(value):
            self._scan_value(child, label)


def harvest_ocatari(ocatari_src: Optional[str], module_basename: str) -> list[dict]:
    """AST-parse ``<ocatari_src>/ram/<module_basename>.py`` -> list of ref dicts.
    Returns [] if the source is unavailable (recorded by caller, no fabrication)."""
    if not ocatari_src:
        return []
    path = os.path.join(ocatari_src, "ram", f"{module_basename}.py")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    v = _OcatariVisitor()
    v.visit(tree)

    # dedupe by (ram_index, offset, concept, slice); keep deterministic order
    seen: dict[tuple, OcatariRef] = {}
    for r in v.refs:
        if r.ram_index is None or not (0 <= r.ram_index < SPEC_RAM_BYTES):
            continue
        key = (r.ram_index, r.offset, r.concept, r.is_slice, r.slice_end)
        seen.setdefault(key, r)
    out = []
    for r in sorted(seen.values(), key=lambda x: (x.ram_index, str(x.concept))):
        out.append(asdict(r))
    return out


# --------------------------------------------------------------------------- #
# (3) Merge into per-game candidate records (SPEC §R schema)
# --------------------------------------------------------------------------- #

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_candidates(game_id: str, ocatari_src: Optional[str]) -> dict:
    ari_key, oc_name = GAME_SRC_NAMES[game_id]
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    candidates: list[dict] = []

    # --- AtariARI candidates (raw, no offset) ---
    ari = ATARIARI_DICT.get(ari_key, {})
    for concept, ram_index in sorted(ari.items(), key=lambda kv: kv[1]):
        candidates.append({
            "concept": concept,
            "ram_index": int(ram_index),
            "ram_addr_hex": f"0x{0x80 + int(ram_index):02X}",  # console RAM mirrors at $80
            "offset": None,             # AtariARI does NOT offset-correct
            "offset_applied": False,
            "raw": True,
            "source": "AtariARI",
            "source_ref": ATARIARI_REF,
            "verified": False,          # correlational candidate (E2-2 verifies)
            "note": "raw RAM index; not aligned to rendered framebuffer position",
        })

    # --- OCAtari candidates (offsets recovered from source) ---
    oc_refs = harvest_ocatari(ocatari_src, oc_name)
    for r in oc_refs:
        off = r["offset"]
        candidates.append({
            "concept": r["concept"],
            "ram_index": int(r["ram_index"]),
            "ram_addr_hex": f"0x{0x80 + int(r['ram_index']):02X}",
            "offset": off,
            "offset_applied": off is not None,
            "raw": off is None,
            "is_slice": bool(r["is_slice"]),
            "slice_end": r["slice_end"],
            "source": "OCAtari",
            "source_ref": OCATARI_REF,
            "extractor_func": r["func"],
            "verified": False,
            "note": ("offset-corrected to rendered position"
                     if off is not None else
                     "RAM-mode extractor reference (no additive offset at use site)"),
        })

    ocatari_available = ocatari_src is not None and bool(oc_refs)

    record = {
        # --- SPEC §R self-describing header ---
        "paper": PAPER,
        "phase": PHASE,
        "method": METHOD,
        "game": game_id,
        "frame": None,
        "state": None,
        "target_output": "ram_to_concept_candidate_map",
        "metric_name": "n_candidate_labels",
        "value": len(candidates),
        "ci": None,
        "stderr": None,
        "n": len(candidates),
        "seed": None,
        "where": "local",
        "commit": _git_commit(),
        "oracle_ref": None,                 # verified by intervention in P2-E2-2
        "timestamp": now,
        # --- payload ---
        "sources": {
            "AtariARI": {
                "available": ari_key in ATARIARI_DICT,
                "ref": ATARIARI_REF,
                "n": len(ari),
                "offset_corrected": False,
            },
            "OCAtari": {
                "available": ocatari_available,
                "ref": OCATARI_REF,
                "src_dir": ocatari_src,
                "module": f"ram/{oc_name}.py",
                "n": len(oc_refs),
                "offset_corrected": True,
            },
        },
        "ram_space": {"size_bytes": SPEC_RAM_BYTES, "base_addr": "0x80"},
        "status": "candidate_correlational",
        "verify_with": "P2-E2-2 (perturb byte -> object moves on exact framebuffer)",
        "candidates": candidates,
    }
    return record


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ocatari-src",
        default=_autodetect_ocatari_src(),
        help="path to the installed ocatari package dir (the one containing ram/). "
             "If omitted, auto-detected; if not found, AtariARI-only candidates "
             "are emitted (recorded as OCAtari unavailable).",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"),
        help="output directory for candidates_<game>.json",
    )
    ap.add_argument("--games", nargs="*", default=CORE_GAMES,
                    help="game ids to emit (default: the 6 core games)")
    args = ap.parse_args(argv)

    src = args.ocatari_src
    if src and not os.path.isdir(os.path.join(src, "ram")):
        print(f"[warn] ocatari src has no ram/ subdir: {src} -> OCAtari unavailable",
              file=sys.stderr)
        src = None
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for game in args.games:
        if game not in GAME_SRC_NAMES:
            print(f"[skip] unknown game id: {game}", file=sys.stderr)
            continue
        rec = build_candidates(game, src)
        out_path = os.path.join(args.out_dir, f"candidates_{game}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, sort_keys=False)
            fh.write("\n")
        n_ari = rec["sources"]["AtariARI"]["n"]
        n_oc = rec["sources"]["OCAtari"]["n"]
        print(f"[ok] {game:14s} -> {os.path.relpath(out_path)}  "
              f"(AtariARI={n_ari}, OCAtari={n_oc}, total={rec['n']})")
        summary.append((game, n_ari, n_oc, rec["n"]))

    print("\nsummary (game, AtariARI, OCAtari, total candidates):")
    for g, a, o, t in summary:
        print(f"  {g:14s} {a:3d} {o:3d} {t:3d}")
    if src is None:
        print("\n[note] OCAtari source unavailable -> AtariARI-only candidates "
              "(no fabrication). Install ocatari and re-run for offset-corrected "
              "OCAtari candidates.", file=sys.stderr)
    return 0


def _autodetect_ocatari_src() -> Optional[str]:
    """Find the installed ocatari dir without importing it (import needs ALE)."""
    for p in sys.path:
        cand = os.path.join(p, "ocatari")
        if os.path.isdir(os.path.join(cand, "ram")):
            return cand
    # fall back to the jaxtari venv site-packages if running under a different python
    try:
        import sysconfig
        cand = os.path.join(sysconfig.get_paths()["purelib"], "ocatari")
        if os.path.isdir(os.path.join(cand, "ram")):
            return cand
    except Exception:
        pass
    return None


if __name__ == "__main__":
    raise SystemExit(main())
