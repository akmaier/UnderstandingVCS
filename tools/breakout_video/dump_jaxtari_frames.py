"""Run jaxtari on Breakout with a given action sequence and dump
per-frame screens as a flat `(n_frames, 210, 160)` uint8 binary
file.

Task #53 (vertical-alignment fix): output shape is `(n, 210, 160)`,
not the old `(n, 192, 160)`. `env.get_screen()` now returns the
ALE-standard `Display.YStart=34` / `Display.Height=210` crop (same as
xitari), so the per-frame screen vertically aligns with
`dump_xitari_frames.py`'s output. The video composer
(`render_breakout_compare.py`) accordingly stopped cropping
xitari to 192 lines from row 18.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.breakout import BreakoutRomSettings
from jaxtari.games.pong import PongRomSettings
from jaxtari.games.atari_classics import (
    PitfallRomSettings, EnduroRomSettings, SeaquestRomSettings,
    BeamriderRomSettings,
)
from jaxtari.games.joystick_starts import (
    AirRaidRomSettings, AsterixRomSettings, DoubleDunkRomSettings,
    ElevatorActionRomSettings, GopherRomSettings, GravitarRomSettings,
    JourneyEscapeRomSettings, PrivateEyeRomSettings, SkiingRomSettings,
    UpNDownRomSettings, YarsRevengeRomSettings,
    # Cluster B (#127b sprint 3+4) — terminal/auto-reset in this auto-resetting
    # video pipeline. asterix already imported; pooyan/pacman extended in place;
    # phoenix new. (ms_pacman is in more_games — see below.)
    PacmanRomSettings, PooyanRomSettings, PhoenixRomSettings,
)
# Cluster B (#127b) — long-horizon terminal/auto-reset. Without a per-game
# RomSettings with a real game-over reader these ROMs fall back to bare
# StellaEnvironment (is_terminal always False), so this auto-resetting video
# pipeline NEVER restarts at game-over while xitari's trace_dump --auto-reset
# does, leaving the dead/old episode rendering (the user-visible long-horizon
# divergence: space_invaders f1092 background flash, road_runner f765,
# kangaroo f1720, asteroids). Mirror of jutari tools/breakout_video/
# dump_jutari_frames.jl (which registers all four here).
from jaxtari.games.space_invaders import SpaceInvadersRomSettings
from jaxtari.games.more_games import (
    AsteroidsRomSettings, RoadRunnerRomSettings, KangarooRomSettings,
    BerzerkRomSettings, MontezumaRevengeRomSettings, RiverRaidRomSettings,
    MsPacmanRomSettings,
)

# ROM basename → RomSettings constructor. Defaults to bare StellaEnvironment
# (no per-game scoring/termination) for unrecognised ROMs — same convention
# as `tools/jutari_screen_dump.jl::_SETTINGS_BY_BASENAME`.
# Pitfall + Enduro override starting_actions() to emulate xitari's
# `getStartingActions()` (UP / FIRE) so frame 0 RAM matches xitari
# (tasks #81/#82).
_SETTINGS_BY_BASENAME = {
    "breakout.bin": BreakoutRomSettings,
    "pong.bin":     PongRomSettings,
    "pitfall.bin":  PitfallRomSettings,
    "enduro.bin":   EnduroRomSettings,
    "seaquest.bin": SeaquestRomSettings,
    # Task #101 — getStartingActions-only joystick games (mirror of jutari).
    "air_raid.bin":        AirRaidRomSettings,
    "asterix.bin":         AsterixRomSettings,
    "beam_rider.bin":      BeamriderRomSettings,
    "double_dunk.bin":     DoubleDunkRomSettings,
    "elevator_action.bin": ElevatorActionRomSettings,
    "gopher.bin":          GopherRomSettings,
    "gravitar.bin":        GravitarRomSettings,
    "journey_escape.bin":  JourneyEscapeRomSettings,
    "private_eye.bin":     PrivateEyeRomSettings,
    "skiing.bin":          SkiingRomSettings,
    "up_n_down.bin":       UpNDownRomSettings,
    "yars_revenge.bin":    YarsRevengeRomSettings,
    # Cluster B (#127b) — terminal/auto-reset so the auto-resetting video
    # pipeline restarts at game-over (mirror of dump_jutari_frames.jl). All
    # four registered here (asteroids included: this dumper auto-resets every
    # iteration on its boot/attract-mode game-over byte, matching xitari's
    # trace_dump --auto-reset attract stall — see jutari note).
    "space_invaders.bin":  SpaceInvadersRomSettings,
    "asteroids.bin":       AsteroidsRomSettings,
    "road_runner.bin":     RoadRunnerRomSettings,
    "kangaroo.bin":        KangarooRomSettings,
    # Cluster B (#127b sprint 3+4) — the remaining long-horizon terminal games
    # (mirror of dump_jutari_frames.jl): berzerk f581, montezuma f867, riverraid
    # f958, asterix f1160, pooyan f1532, phoenix f1743, pacman f1771, ms_pacman
    # f1786 — all die long after the in-window sweep, so registering is safe.
    "berzerk.bin":           BerzerkRomSettings,
    "montezuma_revenge.bin": MontezumaRevengeRomSettings,
    "riverraid.bin":         RiverRaidRomSettings,
    "phoenix.bin":           PhoenixRomSettings,
    "asterix.bin":           AsterixRomSettings,
    "pooyan.bin":            PooyanRomSettings,
    "pacman.bin":            PacmanRomSettings,
    "ms_pacman.bin":         MsPacmanRomSettings,
}


def _load_actions(path: Path) -> list[int]:
    out: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            out.append(int(line))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rom', required=True, type=Path)
    p.add_argument('--actions', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--max-frames', required=True, type=int)
    args = p.parse_args(argv)

    rom = np.fromfile(args.rom, dtype=np.uint8)
    # Per-ROM RomSettings auto-selection by basename. For paddle ROMs
    # (breakout, pong) StellaEnvironment auto-translates LEFT/RIGHT
    # actions into INPT0 dump-pot paddle-position changes — same shape
    # as xitari's `m_use_paddles` / `applyActionPaddles`.
    factory = _SETTINGS_BY_BASENAME.get(args.rom.name)
    env = StellaEnvironment(rom, factory()) if factory else StellaEnvironment(rom)
    # Match the ALE / xitari boot burn so frame 1 alignment matches
    # `tools/trace_dump`'s output.
    env.reset(boot_noop_steps=60, boot_reset_steps=4)

    actions = _load_actions(args.actions)
    n = min(args.max_frames, len(actions))

    # Auto-reset matches xitari's trace_dump --auto-reset default: when
    # the game declares `gameOver=true` (e.g. breakout after losing all
    # 5 lives), re-call env.reset so the next step starts a fresh
    # episode. Mirrors xitari/ale's resetGame() — needed so the
    # comparison video doesn't freeze at game-over while xitari keeps
    # rendering a fresh game.
    frames = np.empty((n, 210, 160), dtype=np.uint8)
    for i in range(n):
        if env.game_over():
            env.reset(boot_noop_steps=60, boot_reset_steps=4)
        env.step(int(actions[i]))
        frames[i] = np.asarray(env.get_screen(), dtype=np.uint8)
        if (i + 1) % 300 == 0:
            print(f"  jaxtari: {i + 1}/{n} frames", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames.tofile(args.out)
    # Shape sidecar read by render_breakout_compare.py::_load_raw. jaxtari is
    # currently NTSC-only (210); on a PAL game the renderer crops both panels
    # to the common height.
    Path(str(args.out) + ".shape").write_text(
        f"{n} {frames.shape[1]} {frames.shape[2]}\n")
    print(f"wrote {n} frames of shape {frames.shape[1:]} to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
