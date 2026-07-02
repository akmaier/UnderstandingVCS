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

# The 54 T3-labeled games (== XAI_LABELED in tools/xai_study/common/game_sets.jl;
# OCAtari-covered, ALE-canonical ids). This is the full set import_labels now
# emits candidates for (previously only the 6 core). EXCLUDED (10 unlabeled):
# defender elevator_action gravitar journey_escape solaris surround tutankham
# videochess wizard_of_wor zaxxon.
XAI_LABELED = [
    "air_raid", "alien", "amidar", "assault", "asterix", "asteroids", "atlantis",
    "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling", "boxing",
    "breakout", "carnival", "centipede", "chopper_command", "crazy_climber",
    "demon_attack", "double_dunk", "enduro", "fishing_derby", "freeway",
    "frostbite", "gopher", "hero", "ice_hockey", "jamesbond", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game", "pacman",
    "phoenix", "pitfall", "pong", "pooyan", "private_eye", "qbert", "riverraid",
    "road_runner", "robotank", "seaquest", "skiing", "space_invaders",
    "star_gunner", "tennis", "time_pilot", "up_n_down", "venture", "video_pinball",
    "yars_revenge",
]

# The 22 games with an AtariARI atari_dict (verbatim below). AtariARI keys use
# the same ALE-canonical id we use, so the AtariARI key == our id for these.
ATARIARI_GAMES = [
    "asteroids", "berzerk", "bowling", "boxing", "breakout", "demon_attack",
    "freeway", "frostbite", "hero", "montezuma_revenge", "ms_pacman", "pitfall",
    "pong", "private_eye", "qbert", "riverraid", "seaquest", "space_invaders",
    "tennis", "venture", "video_pinball", "yars_revenge",
]

# id -> (AtariARI atari_dict key, ocatari ram module basename).
# The ocatari module basename is the id with underscores stripped (verified
# against jaxtari/.venv/.../ocatari/ram/*.py — all 54 resolve, e.g.
# space_invaders->spaceinvaders, ms_pacman->mspacman, beam_rider->beamrider,
# up_n_down->upndown, road_runner->roadrunner, yars_revenge->yarsrevenge).
GAME_SRC_NAMES = {
    game: (game if game in ATARIARI_GAMES else None, game.replace("_", ""))
    for game in XAI_LABELED
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
#     All 22 AtariARI games are transcribed here; values are RAM byte indices
#     0..127. Games in XAI_LABELED with no AtariARI dict get OCAtari-only
#     candidates (recorded as AtariARI unavailable — no fabrication).
# --------------------------------------------------------------------------- #

ATARIARI_DICT: dict[str, dict[str, int]] = {
    "asteroids": {
        "player_x": 73,
        "player_y": 74,
        "player_missile_x1": 83,
        "player_missile_x2": 84,
        "player_missile_y1": 86,
        "player_missile_y2": 87,
        "num_lives_direction": 60,
        "player_score_high": 61,
        "player_score_low": 62,
        "player_missile_direction": 89,
    },
    "berzerk": {
        "player_x": 19,
        "player_y": 11,
        "player_direction": 14,
        "player_missile_x": 22,
        "player_missile_y": 23,
        "player_missile_direction": 21,
        "robot_missile_direction": 26,
        "robot_missile_x": 29,
        "robot_missile_y": 30,
        "num_lives": 90,
        "robots_killed_count": 91,
        "game_level": 92,
        "enemy_evilOtto_x": 46,
        "enemy_evilOtto_y": 89,
        "enemy_robots_x": range(65, 73),  # 65-72
        "player_score": range(93, 96),  # 93, 94, 95
    },
    "bowling": {
        "ball_x": 30,
        "ball_y": 41,
        "player_x": 29,
        "player_y": 40,
        "frame_number_display": 36,
        "pin_existence": range(57, 67),  # 57-66
        "score": 33,
    },
    "boxing": {
        "player_x": 32,
        "player_y": 34,
        "enemy_x": 33,
        "enemy_y": 35,
        "enemy_score": 19,
        "clock": 17,
        "player_score": 18,
    },
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
    "demon_attack": {
        "level": 62,
        "player_x": 22,
        "enemy_x1": 17,
        "enemy_x2": 18,
        "enemy_x3": 19,
        "missile_y": 21,
        "enemy_y1": 69,
        "enemy_y2": 70,
        "enemy_y3": 71,
        "num_lives": 114,
    },
    "freeway": {
        "player_y": 14,
        "score": 103,
        "enemy_car_x": range(108, 118),  # 108-117
    },
    "frostbite": {
        "top_row_iceflow_x": 34,
        "second_row_iceflow_x": 33,
        "third_row_iceflow_x": 32,
        "fourth_row_iceflow_x": 31,
        "enemy_bear_x": 104,
        "num_lives": 76,
        "igloo_blocks_count": 77,  # 255 is none and 15 is all "

        "enemy_x": range(84, 88),  # 84, 85, 86, 87
        "player_x": 102,
        "player_y": 100,
        "player_direction": 4,
        "score": range(72, 75),  # 72, 73, 74
    },
    "hero": {
        "player_x": 27,
        "player_y": 31,
        "power_meter": 43,
        "room_number": 28,
        "level_number": 117,
        "dynamite_count": 50,
        "score": range(56, 61),
    },
    "montezuma_revenge": {
        "room_number": 3,
        "player_x": 42,
        "player_y": 43,
        "player_direction": 52,  # 72: facing left, 128: facing right, 0: facing left, 32: facing right ???
        "enemy_skull_x": 47,
        "enemy_skull_y": 46,
        "key_monster_x": 44,
        "key_monster_y": 45,
        "level": 57,
        "num_lives": 58,
        "items_in_inventory_count": 61,
        "room_state": 62,
        "score_0": 19,
        "score_1": 20,
        "score_2": 21,
    },
    "pitfall": {
        "player_x": 97,  # 8-148
        "player_y": 105,  # 21-86 except for when respawning then 0-255 with confusing wraparound
        "enemy_logs_x": 98,  # 0-160
        "enemy_scorpion_x": 99,
        # "player_y_on_ladder": 108, # 0-20
        # "player_collided_with_rope": 5, #yes if bit 6 is 1
        "bottom_of_rope_x": 18,  # tells you which section they are in 0-255 it channels 8 sections repeatedly
    },
    "private_eye": {
        "player_x": 63,
        "player_y": 86,
        "room_number": 92,
        "clock": range(67, 69),  # 67, 68
        "player_direction": 58,
        "score": range(77, 80),  # 77, 78, 79
        "dove_x": 48,
        "dove_y": 39,
    },
    "riverraid": {
        "player_x": 51,
        "missile_x": 117,
        "missile_y": 50,
        "fuel_meter_high": 55,  # high value displayed
        "fuel_meter_low": 56,  # low value
    },
    "tennis": {
        "enemy_x": 27,
        "enemy_y": 25,
        "enemy_score": 70,
        "ball_x": 16,
        "ball_y": 17,
        "player_x": 26,
        "player_y": 24,
        "player_score": 69,
    },
    "venture": {
        "sprite0_y": 20,
        "sprite1_y": 21,
        "sprite2_y": 22,
        "sprite3_y": 23,
        "sprite4_y": 24,
        "sprite5_y": 25,
        "sprite0_x": 79,
        "sprite1_x": 80,
        "sprite2_x": 81,
        "sprite3_x": 82,
        "sprite4_x": 83,
        "sprite5_x": 84,
        "player_x": 85,
        "player_y": 26,
        "current_room": 90,  # The number of the room the player is currently in 0 to 9_
        "num_lives": 70,
        "score_1_2": 71,
        "score_3_4": 72,
    },
    "video_pinball": {
        "ball_x": 67,
        "ball_y": 68,
        "player_left_paddle_y": 98,
        "player_right_paddle_y": 102,
        "score_1": 48,
        "score_2": 50,
    },
    "yars_revenge": {
        "player_x": 32,
        "player_y": 31,
        "player_missile_x": 38,
        "player_missile_y": 37,
        "enemy_x": 43,
        "enemy_y": 42,
        "enemy_missile_x": 47,
        "enemy_missile_y": 46,
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
    # A concept's value is a single RAM index, OR a range/list of indices for a
    # multi-byte concept (e.g. berzerk enemy_robots_x = range(65,73)). Multi-byte
    # concepts are expanded into per-byte candidates concept[k] so each byte gets
    # its own intervention test. ari_key is None for OCAtari-only games -> no dict.
    ari = ATARIARI_DICT.get(ari_key, {}) if ari_key is not None else {}
    ari_flat: list[tuple[str, int]] = []
    for concept, val in ari.items():
        if isinstance(val, (range, list, tuple)):
            for k, idx in enumerate(val):
                ari_flat.append((f"{concept}[{k}]", int(idx)))
        else:
            ari_flat.append((concept, int(val)))
    for concept, ram_index in sorted(ari_flat, key=lambda kv: (kv[1], kv[0])):
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
                "available": ari_key is not None and ari_key in ATARIARI_DICT,
                "ref": ATARIARI_REF,
                "n": len(ari_flat),   # per-byte count (multi-byte concepts expanded)
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
    ap.add_argument("--games", nargs="*", default=XAI_LABELED,
                    help="game ids to emit (default: the 54 XAI_LABELED games)")
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
