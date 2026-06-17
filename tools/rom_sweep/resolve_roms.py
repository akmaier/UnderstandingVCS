#!/usr/bin/env python3
"""Resolve the 64 xitari-supported ALE games to ROM files in the collection.

xitari/ALE selects a game purely by ROM *filename* basename == RomSettings.rom()
(see xitari/games/Roms.cpp::buildRomRLWrapper). So to run game <g> we need a file
named <g>.bin that is a valid ROM. We match each canonical rom() name against the
5111-file collection by normalized title (lowercase, strip the trailing
"(...)" descriptors, drop non-alphanumerics), preferring NTSC originals over
PAL/Prototype/Beta/hack dumps, and copy the winner to tools/rom_sweep/roms/<g>.bin.

Conformance note: both jutari and xitari run the *same bytes*, so a slightly
imperfect title match does not invalidate the RAM/screen conformance measurement
(both emulate identical ROM bytes); it only affects the per-game label + which
xitari RomSettings auto-applies. We still try to match correctly for clean
reporting.
"""
from __future__ import annotations
import re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAMES = (REPO / "tools/rom_sweep/rom_names.txt").read_text().split()
# The ROM source folder is xitari/roms (the in-repo 2207-bin collection with
# descriptive filenames). Per-game canonical names are fuzzy-matched against it;
# the 6 already-canonical files (breakout.bin, pong.bin, ...) are picked directly
# via the CURATED shortcut below.
COLLECTION = REPO / "xitari/roms"
CURATED = REPO / "xitari/roms"
OUT = REPO / "tools/rom_sweep/roms"
OUT.mkdir(parents=True, exist_ok=True)

# Explicit overrides for games whose canonical name can't fuzzy-match the
# collection's verbose title: "Montezuma's Revenge" norms to `montezumas…`
# (the possessive breaks the prefix), "Robot Tank" norms to `robottank`
# (double-t vs canonical `robotank`). Exact NTSC-original filenames in xitari/roms.
OVERRIDES = {
    "montezuma_revenge": "Montezuma's Revenge - Featuring Panama Joe (Monty) (1984) (Parker Brothers - JWDA, Henry Will IV) (PB5760) ~.bin",
    "robotank": "Robot Tank (Robotank) (1983) (Activision, Alan Miller) (AZ-028, AG-028-04) ~.bin",
}

def norm(s: str) -> str:
    # take the leading title (before first parenthesis group), drop non-alnum
    s = s.split("(")[0]
    return re.sub(r"[^a-z0-9]", "", s.lower())

def penalty(fn: str) -> int:
    low = fn.lower()
    p = 0
    # Region / dump-quality penalties.
    for bad, w in (("pal", 5), ("prototype", 4), ("beta", 3), ("hack", 6),
                   ("secam", 5), ("(e)", 2), ("(g)", 2), ("(f)", 2),
                   ("genesis", 8), ("unknown", 1)):
        if bad in low:
            p += w
    # Clone / pirate / multicart distributor penalties — prefer the canonical
    # NTSC release (Atari/Activision/Parker Bros/Sega/Imagic/...) over these
    # Brazilian-clone / multicart dumps that the fuzzy match otherwise picks.
    for clone, w in (("dactari", 9), ("milmar", 9), ("fotomania", 9),
                     ("bit corp", 9), ("bitcorp", 9), ("32 in 1", 9),
                     ("4 in 1", 9), ("4 game in one", 9), ("zellers", 7),
                     ("panda", 7), ("puzzy", 9), ("rentacom", 9),
                     ("digivision", 9), ("digitel", 9), ("gamegear", 9),
                     ("cce", 7), ("vdi", 7), ("home vision", 7),
                     ("hitech", 7), ("quelle", 7), ("zirok", 9),
                     ("supergame", 9), ("dynacom", 9), ("rad action", 9),
                     ("eskimo jump", 9), ("ariola", 6)):
        if clone in low:
            p += w
    # Bonus (negative penalty) for a recognized original publisher — breaks
    # penalty ties in favour of the canonical NTSC release rather than the
    # shortest (often clone) filename.
    for pub in ("activision", "(atari", "parker bros", "imagic", "sega",
                "coleco", "mattel", "cbs", "20th century fox", "konami"):
        if pub in low:
            p -= 1
            break
    # Prefer dumps that look canonical (have a 4-digit 197x/198x year).
    import re as _re
    if not _re.search(r"\((19[78]\d)\)", fn):
        p += 1
    return p

# Index the collection by normalized title.
cands: dict[str, list[Path]] = {}
for f in COLLECTION.rglob("*.bin"):
    cands.setdefault(norm(f.name), []).append(f)
all_files = [f for fs in cands.values() for f in fs]

def best_for(game: str) -> Path | None:
    g = game.replace("_", "")
    # 1) curated clean ROM already present under the canonical name?
    # 2) exact normalized-title equality; 3) startswith; 4) contains.
    pools = []
    exact = [f for f in all_files if norm(f.name) == g]
    starts = [f for f in all_files if norm(f.name).startswith(g) and f not in exact]
    contains = [f for f in all_files if g in norm(f.name) and f not in exact and f not in starts]
    for pool in (exact, starts, contains):
        if pool:
            return sorted(pool, key=lambda f: (penalty(f.name), len(f.name)))[0]
    return None

resolved, unresolved = {}, []
for g in NAMES:
    # Explicit override first (verbose titles the fuzzy matcher misses).
    ov = OVERRIDES.get(g)
    if ov and (COLLECTION / ov).is_file():
        shutil.copy(COLLECTION / ov, OUT / f"{g}.bin")
        resolved[g] = f"(override) {ov}"
        continue
    # Prefer an already-curated, correctly-named ROM if one exists.
    curated = CURATED / f"{g}.bin"
    if curated.is_file():
        shutil.copy(curated, OUT / f"{g}.bin"); resolved[g] = f"(curated) {curated.name}"; continue
    f = best_for(g)
    if f is None:
        unresolved.append(g); continue
    shutil.copy(f, OUT / f"{g}.bin"); resolved[g] = f.name

print(f"resolved {len(resolved)}/{len(NAMES)};  unresolved {len(unresolved)}")
manifest = REPO / "tools/rom_sweep/manifest.txt"
with open(manifest, "w") as m:
    for g in NAMES:
        m.write(f"{g}\t{resolved.get(g, 'UNRESOLVED')}\n")
print("UNRESOLVED:", " ".join(unresolved) if unresolved else "(none)")
print(f"manifest -> {manifest}")
