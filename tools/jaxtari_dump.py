#!/usr/bin/env python3
"""Isolated single jaxtari run — dump per-frame RAM or SCREEN to a binary file.

This is the jaxtari analogue of `jutari_trace_dump.jl` / `jutari_screen_dump.jl`.
It is the per-job worker the parallel sweep (`tools/rom_sweep/sweep_jaxtari.py`)
shells out to: ONE ROM, ONE process, so JAX state is fully isolated and the
sweep can run many of these concurrently without JAX thread-safety hazards.

THREADING: jaxtari runs eager on the CPU with tiny (160-element) arrays, so
XLA/BLAS multi-threading only adds overhead. We pin this process SINGLE-THREADED
(below, before importing jax) so the sweep can run N = n_cores of these in
parallel and saturate the hardware without oversubscription. The driver also
sets these in the child env; the setdefault here is a standalone-run fallback.

Output: raw little-endian uint8 bytes, frames concatenated:
  - mode=ram    : 128 bytes/frame   (RIOT RAM)
  - mode=screen : h*160 bytes/frame (h = env.get_screen() height; 210 NTSC)

Usage:
    jaxtari/.venv/bin/python tools/jaxtari_dump.py \\
        --rom xitari/roms/pong.bin --actions <actions.txt> \\
        --max-frames 30 --mode ram --out /tmp/pong_ram.bin
"""
from __future__ import annotations

import os

# --- pin single-threaded BEFORE importing jax/jaxtari -----------------------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Disable XLA CPU Eigen multi-threading (one op = one thread) so each worker
# stays on a single core.
_xla = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _xla:
    os.environ["XLA_FLAGS"] = (_xla + " --xla_cpu_multi_thread_eigen=false").strip()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "jaxtari"))

from jaxtari.env.stella_environment import StellaEnvironment  # noqa: E402
# Import every RomSettings class from the top-level package, which re-exports
# the canonical class for each name (the submodules have duplicates / differ in
# location — e.g. QbertRomSettings lives in more_games, not joystick_starts).
from jaxtari.games import (  # noqa: E402
    GenericRomSettings, RomSettings,
    BreakoutRomSettings, PongRomSettings,
    PitfallRomSettings, EnduroRomSettings, SeaquestRomSettings,
    BeamriderRomSettings,
    AirRaidRomSettings, AmidarRomSettings, AsterixRomSettings,
    BattleZoneRomSettings, CarnivalRomSettings, DoubleDunkRomSettings,
    ElevatorActionRomSettings, GopherRomSettings, GravitarRomSettings,
    JourneyEscapeRomSettings, MsPacmanRomSettings, PacmanRomSettings,
    PooyanRomSettings, PrivateEyeRomSettings, SkiingRomSettings,
    SurroundRomSettings, UpNDownRomSettings, YarsRevengeRomSettings,
    QbertRomSettings, WizardOfWorRomSettings,
    # Cluster B (#127b): long-horizon terminal-detection. SpaceInvaders +
    # Asteroids already had settings classes but were never registered here;
    # RoadRunner + Kangaroo are new (mirror xitari RoadRunner.cpp/Kangaroo.cpp).
    SpaceInvadersRomSettings, AsteroidsRomSettings,
    RoadRunnerRomSettings, KangarooRomSettings,
    # Cluster B (#127b sprint 3+4): the remaining long-horizon terminal/auto-
    # reset games. berzerk/montezuma_revenge/riverraid/phoenix are new classes;
    # asterix/pooyan/pacman/ms_pacman already registered above now carry a real
    # terminal reader (predicate added in-place, render props preserved).
    BerzerkRomSettings, MontezumaRevengeRomSettings, RiverRaidRomSettings,
    PhoenixRomSettings,
)

# Full 25-game per-ROM RomSettings map — matches jutari's
# `_SETTINGS_BY_BASENAME` in tools/jutari_trace_dump.jl (the GOLDEN reference)
# so the jaxtari sweep is apples-to-apples. This is a SUPERSET of
# tools/check_trace.py's 19-game map: check_trace omits battle_zone, carnival,
# ms_pacman, pacman, pooyan, qbert (those run generic there) — which would be a
# settings-vs-emulation false divergence in the sweep, so we add them here.
_SETTINGS_BY_BASENAME: dict[str, type[RomSettings]] = {
    "breakout.bin": BreakoutRomSettings,
    "pong.bin": PongRomSettings,
    "pitfall.bin": PitfallRomSettings,
    "enduro.bin": EnduroRomSettings,
    "seaquest.bin": SeaquestRomSettings,
    "air_raid.bin": AirRaidRomSettings,
    "amidar.bin": AmidarRomSettings,
    "asterix.bin": AsterixRomSettings,
    "beam_rider.bin": BeamriderRomSettings,
    "double_dunk.bin": DoubleDunkRomSettings,
    "elevator_action.bin": ElevatorActionRomSettings,
    "gopher.bin": GopherRomSettings,
    "gravitar.bin": GravitarRomSettings,
    "journey_escape.bin": JourneyEscapeRomSettings,
    "private_eye.bin": PrivateEyeRomSettings,
    "skiing.bin": SkiingRomSettings,
    "surround.bin": SurroundRomSettings,
    "up_n_down.bin": UpNDownRomSettings,
    "yars_revenge.bin": YarsRevengeRomSettings,
    "battle_zone.bin": BattleZoneRomSettings,
    "carnival.bin": CarnivalRomSettings,
    "ms_pacman.bin": MsPacmanRomSettings,
    "pacman.bin": PacmanRomSettings,
    "pooyan.bin": PooyanRomSettings,
    "qbert.bin": QbertRomSettings,
    "wizard_of_wor.bin": WizardOfWorRomSettings,
    # Cluster B (#127b): terminal/auto-reset gap on long-horizon rollouts.
    # These ROMs previously fell back to GenericRomSettings (is_terminal always
    # False) so the comparison-video pipeline never auto-reset at game-over,
    # while xitari/ALE does. Registering a real terminal reader closes the gap.
    "space_invaders.bin": SpaceInvadersRomSettings,
    "asteroids.bin": AsteroidsRomSettings,
    "road_runner.bin": RoadRunnerRomSettings,
    "kangaroo.bin": KangarooRomSettings,
    # Cluster B (#127b sprint 3+4): remaining long-horizon terminal games. The
    # 8 die LONG after the 30-60 frame sweep window (berzerk f581, montezuma
    # f867, riverraid f958, asterix f1160, phoenix f1743, pacman f1771,
    # ms_pacman f1786, pooyan f1532) so registering them cannot freeze the
    # non-resetting in-window sweep (verified non-terminal through frame 60).
    "berzerk.bin": BerzerkRomSettings,
    "montezuma_revenge.bin": MontezumaRevengeRomSettings,
    "riverraid.bin": RiverRaidRomSettings,
    "phoenix.bin": PhoenixRomSettings,
}


def _settings_for_rom(rom_path: Path) -> RomSettings:
    cls = _SETTINGS_BY_BASENAME.get(rom_path.name, GenericRomSettings)
    return cls()


def _load_actions(path: Path) -> list[int]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(int(s))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True)
    p.add_argument("--max-frames", type=int, required=True)
    p.add_argument("--mode", choices=("ram", "screen"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--construction-probe", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="run xitari's double-boot construction probe "
                        "(ALEInterface ctor format-autodetect + double reset_game). "
                        "xitari's trace_dump does this natively and jutari's "
                        "jutari_trace_dump.jl defaults to it — so this DEFAULTS TRUE "
                        "for full xitari/jutari parity (seeds surround's $7d boot "
                        "counter). Pass --no-construction-probe for a single-boot.")
    args = p.parse_args(argv)

    rom = np.frombuffer(args.rom.read_bytes(), dtype=np.uint8)
    settings = _settings_for_rom(args.rom)
    env = StellaEnvironment(rom, settings)
    # xitari's ALEInterface::resetGame() burns 60 NOOP + 4 RESET frames; mirror
    # it so the post-boot state matches the xitari trace_dump reference.
    # `--construction-probe` additionally runs xitari's double-boot construction
    # probe (which trace_dump performs natively via the ALEInterface ctor) — the
    # missing piece that seeds free-running counters / beam phase on the
    # probe-sensitive ROMs (surround, demon_attack, …).
    env.reset(boot_noop_steps=60, boot_reset_steps=4,
              construction_probe=args.construction_probe)

    actions = _load_actions(args.actions)
    n = min(args.max_frames, len(actions))
    chunks: list[bytes] = []
    for a in actions[:n]:
        env.step(int(a))
        if args.mode == "ram":
            arr = np.asarray(env.get_ram(), dtype=np.uint8)  # 128 B
        else:
            arr = np.asarray(env.get_screen(), dtype=np.uint8)  # h x 160
        chunks.append(arr.tobytes())
    args.out.write_bytes(b"".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
