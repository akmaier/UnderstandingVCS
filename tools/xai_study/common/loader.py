"""loader — resolve a game *name* to a ROM and build a jaxtari env for it.

This wraps the working ROM-loading pattern from `tools/jaxtari_dump.py` and the
name->file resolution from `tools/rom_sweep/resolve_roms.py` behind two stable
calls every P2 experiment can rely on:

    rom_path = loader.resolve_rom("pong")          # -> Path to a .bin
    env, rom = loader.load_game("pong")            # -> (StellaEnvironment, rom bytes)

ROM resolution order (first hit wins):
  1. an explicit `rom_path=` argument (used as-is);
  2. a curated, canonically-named file `xitari/roms/<name>.bin`;
  3. a sweep-resolved file `tools/rom_sweep/roms/<name>.bin`;
  4. a fuzzy title match against the full ROM collection (NTSC originals
     preferred over PAL/clone dumps), mirroring `resolve_roms.py`.

ROMs are gitignored and read **in place** — never copied into the worktree,
never committed (SCRUM §7). In a developer worktree the gitignored files live
only in the PRIMARY checkout, so we search there too via `_primary_repo()`.

The per-game `RomSettings` come straight from jaxtari (same map as
`jaxtari_dump.py`); an unknown name falls back to `GenericRomSettings`, which is
correct for the breadth set.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# --- repo / ROM locations ---------------------------------------------------
# tools/xai_study/common/loader.py -> repo root is parents[3].
REPO = Path(__file__).resolve().parents[3]


def _primary_repo() -> Path:
    """Best guess at the PRIMARY (non-worktree) checkout.

    Gitignored ROMs/venv live only in the primary checkout, so a developer
    worktree must read them from there. If we *are* the primary (or can't tell),
    return our own REPO. Allow an explicit override for robustness.
    """
    env = os.environ.get("XAI_PRIMARY_REPO")
    if env and Path(env).is_dir():
        return Path(env)
    # A worktree path looks like <primary>/.claude/worktrees/<name>; the primary
    # is the path before "/.claude/worktrees/".
    s = str(REPO)
    marker = "/.claude/worktrees/"
    if marker in s:
        primary = Path(s.split(marker, 1)[0])
        if primary.is_dir():
            return primary
    return REPO


def _rom_search_roots() -> list[Path]:
    """Directories that may hold ROMs, in priority order, across both checkouts."""
    roots: list[Path] = []
    for base in (REPO, _primary_repo()):
        for sub in ("xitari/roms",
                    "tools/rom_sweep/roms",
                    "xitari/games/Atari-2600-VCS-ROM-Collection/ROMS"):
            p = base / sub
            if p.is_dir() and p not in roots:
                roots.append(p)
    return roots


def _jaxtari_root() -> Path:
    """The jaxtari package dir, preferring the primary checkout's (has its venv)."""
    for base in (_primary_repo(), REPO):
        p = base / "jaxtari"
        if (p / "jaxtari").is_dir():
            return p
    return REPO / "jaxtari"


# --- fuzzy title matching (mirrors resolve_roms.py) -------------------------
_OVERRIDES = {
    "montezuma_revenge": "Montezuma's Revenge",
    "robotank": "Robot Tank",
}


def _norm(s: str) -> str:
    """Normalize a title: leading segment before '(', alphanumerics only."""
    s = s.split("(")[0]
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _penalty(fn: str) -> int:
    """Prefer NTSC originals over PAL / clone / multicart dumps (resolve_roms.py)."""
    low = fn.lower()
    p = 0
    for bad, w in (("pal", 5), ("prototype", 4), ("beta", 3), ("hack", 6),
                   ("secam", 5), ("(e)", 2), ("(g)", 2), ("(f)", 2),
                   ("genesis", 8), ("unknown", 1)):
        if bad in low:
            p += w
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
    for pub in ("activision", "(atari", "parker bros", "imagic", "sega",
                "coleco", "mattel", "cbs", "20th century fox", "konami"):
        if pub in low:
            p -= 1
            break
    if not re.search(r"\((19[78]\d)\)", fn):
        p += 1
    return p


def _fuzzy_find(name: str) -> Optional[Path]:
    """Fuzzy-match `name` against the full collection; None if no candidate."""
    target = _OVERRIDES.get(name, name).replace("_", "")
    target = _norm(target)
    # Only the big collection has verbose titles worth fuzzy-matching.
    collections = [r for r in _rom_search_roots()
                   if "Atari-2600-VCS-ROM-Collection" in str(r)]
    files: list[Path] = []
    for root in collections:
        files.extend(root.rglob("*.bin"))
    if not files:
        return None
    exact = [f for f in files if _norm(f.name) == target]
    starts = [f for f in files if _norm(f.name).startswith(target) and f not in exact]
    contains = [f for f in files
                if target in _norm(f.name) and f not in exact and f not in starts]
    for pool in (exact, starts, contains):
        if pool:
            return sorted(pool, key=lambda f: (_penalty(f.name), len(f.name)))[0]
    return None


def resolve_rom(name: str, rom_path: Optional[os.PathLike] = None) -> Path:
    """Return the ROM file for a game `name` (or echo an explicit `rom_path`).

    Raises FileNotFoundError if nothing resolves, with the roots searched.
    """
    if rom_path is not None:
        p = Path(rom_path)
        if not p.is_file():
            raise FileNotFoundError(f"explicit rom_path not found: {p}")
        return p
    name = name.strip()
    # Curated / sweep-resolved canonical filenames first.
    for root in _rom_search_roots():
        cand = root / f"{name}.bin"
        if cand.is_file():
            return cand
    found = _fuzzy_find(name)
    if found is not None:
        return found
    roots = "\n  ".join(str(r) for r in _rom_search_roots()) or "(none found)"
    raise FileNotFoundError(
        f"could not resolve ROM for game '{name}'. Searched:\n  {roots}\n"
        "Set XAI_PRIMARY_REPO to the primary checkout if running from a worktree, "
        "or pass rom_path=.")


def _settings_for(name: str, rom_path: Path):
    """Per-game jaxtari RomSettings for `name`/`rom_path` (Generic fallback).

    Reuses the basename->class map from `tools/jaxtari_dump.py` so loader and the
    Paper-1 sweeps pick the *same* settings for a given game (apples-to-apples).
    """
    _ensure_jaxtari_on_path()
    import importlib.util

    dump_path = _primary_repo() / "tools" / "jaxtari_dump.py"
    if not dump_path.is_file():
        dump_path = REPO / "tools" / "jaxtari_dump.py"
    cls = None
    if dump_path.is_file():
        spec = importlib.util.spec_from_file_location("_jaxtari_dump", dump_path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mapping = getattr(mod, "_SETTINGS_BY_BASENAME", {})
            cls = mapping.get(f"{name}.bin") or mapping.get(rom_path.name)
        except Exception:
            cls = None
    if cls is None:
        from jaxtari.games import GenericRomSettings as cls  # type: ignore
    return cls()


def _ensure_jaxtari_on_path() -> None:
    """Put the jaxtari package dir on sys.path (idempotent)."""
    root = str(_jaxtari_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_rom_bytes(name: str, rom_path: Optional[os.PathLike] = None) -> Tuple[np.ndarray, Path]:
    """Return (uint8 ROM bytes, resolved path) for a game name."""
    p = resolve_rom(name, rom_path)
    return np.frombuffer(p.read_bytes(), dtype=np.uint8), p


def load_game(name: str,
              rom_path: Optional[os.PathLike] = None,
              *,
              reset: bool = True,
              boot_noop_steps: int = 60,
              boot_reset_steps: int = 4,
              construction_probe: bool = True,
              random_noop_max: int = 0,
              seed: Optional[int] = None):
    """Build a jaxtari `StellaEnvironment` for game `name`.

    Defaults reproduce the xitari/ALE boot convention used by the Paper-1 sweeps
    (`tools/jaxtari_dump.py`): 60 NOOP + 4 RESET frames + the double-boot
    construction probe, so post-boot state matches the xitari reference and lands
    inside the conformance horizon. Set `reset=False` to inspect the pre-reset
    console (tests may want the unreset state).

    Returns `(env, rom_bytes)`. The resolved ROM path is `env`-independent; call
    `resolve_rom(name)` if you need it.
    """
    _ensure_jaxtari_on_path()
    from jaxtari.env.stella_environment import StellaEnvironment

    rom, p = load_rom_bytes(name, rom_path)
    settings = _settings_for(name, p)
    env = StellaEnvironment(rom, settings)
    if reset:
        env.reset(boot_noop_steps=boot_noop_steps,
                  boot_reset_steps=boot_reset_steps,
                  random_noop_max=random_noop_max,
                  seed=seed,
                  construction_probe=construction_probe)
    return env, rom
