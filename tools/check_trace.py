#!/usr/bin/env python3
"""PXC1 conformance harness — replay a JSONL trace against jaxtari.

Reads a JSONL trace produced by `tools/trace_dump` (run against xitari /
ALE) and replays the same action sequence against jaxtari's
`StellaEnvironment`. At each frame the harness compares jaxtari's RAM
(and optionally the screen) against the reference trace's value; the
first divergence is reported, with a byte-level diff of the offending
frame.

Usage
-----

    python tools/check_trace.py \\
        --rom xitari/roms/pong.bin \\
        --trace tools/fixtures/traces/pong_noop_10.jsonl

    # also check the framebuffer (requires the trace to have been
    # generated with --screen):
    python tools/check_trace.py --rom <rom> --trace <trace> --check-screen

Exit code
---------

    0  every frame's RAM (and screen, if checked) matches the reference
    1  divergence — diagnostic on stderr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.rom_settings import GenericRomSettings, RomSettings
from jaxtari.games.breakout import BreakoutRomSettings
from jaxtari.games.pong import PongRomSettings
from jaxtari.games.atari_classics import (
    PitfallRomSettings, EnduroRomSettings, SeaquestRomSettings,
    BeamriderRomSettings,
)
from jaxtari.games.joystick_starts import (
    AirRaidRomSettings, AmidarRomSettings, AsterixRomSettings,
    DoubleDunkRomSettings,
    ElevatorActionRomSettings, GopherRomSettings, GravitarRomSettings,
    JourneyEscapeRomSettings, PrivateEyeRomSettings, SkiingRomSettings,
    UpNDownRomSettings, YarsRevengeRomSettings,
)


def _load_rom(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


# Per-ROM auto-detection of which RomSettings to use. Currently keyed
# on basename — xitari does the same job via stella.pro's MD5 →
# property lookup but we don't ship stella.pro, so a small filename
# table is the pragmatic equivalent. Add a row when you bring a new
# paddle/joystick ROM under conformance test. Default falls back to
# `GenericRomSettings` (joystick).
_SETTINGS_BY_BASENAME: dict[str, type[RomSettings]] = {
    "breakout.bin": BreakoutRomSettings,
    "pong.bin":     PongRomSettings,
    "pitfall.bin":  PitfallRomSettings,
    "enduro.bin":   EnduroRomSettings,
    "seaquest.bin": SeaquestRomSettings,
    # Task #101 — getStartingActions-only joystick games (mirror of jutari).
    "air_raid.bin":        AirRaidRomSettings,
    "amidar.bin":          AmidarRomSettings,
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
}


def _settings_for_rom(rom_path: Path) -> RomSettings:
    cls = _SETTINGS_BY_BASENAME.get(rom_path.name, GenericRomSettings)
    return cls()


def _hex_of(arr: np.ndarray) -> str:
    return ''.join(f'{int(b):02x}' for b in arr)


def _ram_diff_report(reference_hex: str, actual_hex: str,
                     frame: int, action: int, out) -> None:
    ref = bytes.fromhex(reference_hex)
    act = bytes.fromhex(actual_hex)
    diffs = [(i, ref[i], act[i]) for i in range(len(ref)) if ref[i] != act[i]]
    print(f"DIVERGENCE at frame {frame}, action={action}", file=out)
    print(f"  {len(diffs)} of {len(ref)} RAM bytes differ. First 16:", file=out)
    for i, e, a in diffs[:16]:
        print(f"    RAM[${i:02x}]: xitari=${e:02x}  jaxtari=${a:02x}", file=out)


def _cpu_state(env) -> dict:
    """Pull CPU registers out of jaxtari's HARD console for comparison
    against the `cpu` field a `trace_dump --cpu` trace produces."""
    cpu = env.console.cpu
    return {
        "A":  int(cpu.A),
        "X":  int(cpu.X),
        "Y":  int(cpu.Y),
        "SP": int(cpu.SP),
        "P":  int(cpu.P),
        "PC": int(cpu.PC),
    }


def check_trace(rom_path: Path, trace_path: Path,
                check_screen: bool = False) -> int:
    """Run the conformance check. Returns the number of frames matched
    before a divergence (or until the trace ended); raises on divergence
    via ConformanceError.

    If the reference trace was generated with `--cpu`, every frame's
    CPU state (A, X, Y, SP, P, PC) is also diff'd. CPU divergence is
    reported separately from RAM divergence so the cause class is
    visible immediately: matched CPU + mismatched RAM = the two
    emulators executed the same instructions but data-path reads
    differed; mismatched CPU = execution itself diverged.
    """
    rom = _load_rom(rom_path)
    # PXC1-x round 5 — pick the right RomSettings for the ROM so
    # `StellaEnvironment` auto-applies paddle-action handling (and
    # therefore activates the dump-pot model on INPT0/INPT1) for
    # paddle games. Without this jaxtari's INPT reads return a static
    # `0x80` instead of the cycle-dependent value xitari produces,
    # which is one of the documented PXC1 RAM-divergence sources.
    settings = _settings_for_rom(rom_path)
    env = StellaEnvironment(rom, settings)
    # xitari's `ALEInterface::resetGame()` burns 60 NOOP frames + 4 RESET
    # switch frames before the user's first `act()` (see
    # xitari/environment/stella_environment.cpp). The conformance harness
    # must match.
    env.reset(boot_noop_steps=60, boot_reset_steps=4)

    matched = 0
    with open(trace_path) as f:
        for line in f:
            ref = json.loads(line)
            env.step(int(ref['action']))

            # CPU check — only fires when the reference trace carries a
            # "cpu" field (added by `trace_dump --cpu`).
            if 'cpu' in ref:
                got = _cpu_state(env)
                ref_cpu = {k: ref['cpu'][k] for k in ("A", "X", "Y", "SP", "P", "PC")}
                if got != ref_cpu:
                    print(f"CPU DIVERGENCE at frame {ref['frame']}, "
                          f"action={ref['action']}:", file=sys.stderr)
                    for k in ("A", "X", "Y", "SP", "P", "PC"):
                        if got[k] != ref_cpu[k]:
                            print(f"    {k}: xitari={ref_cpu[k]}  "
                                  f"jaxtari={got[k]}", file=sys.stderr)
                    raise ConformanceError(
                        f"CPU divergence at frame {ref['frame']} "
                        f"({matched} matched)")

            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            ram_hex = _hex_of(ram)
            if ram_hex != ref['ram']:
                _ram_diff_report(ref['ram'], ram_hex,
                                 int(ref['frame']), int(ref['action']),
                                 sys.stderr)
                raise ConformanceError(
                    f"RAM divergence at frame {ref['frame']} ({matched} matched)")

            if check_screen and 'screen' in ref:
                screen = np.asarray(env.get_screen(), dtype=np.uint8)
                if _hex_of(screen.ravel()) != ref['screen']:
                    raise ConformanceError(
                        f"screen divergence at frame {ref['frame']} "
                        f"({matched} matched)")

            matched += 1
    return matched


class ConformanceError(AssertionError):
    """Raised when a jaxtari frame diverges from the xitari reference."""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rom', required=True, type=Path)
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--check-screen', action='store_true')
    args = p.parse_args(argv)

    try:
        matched = check_trace(args.rom, args.trace, args.check_screen)
    except ConformanceError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"OK — {matched} frame(s) match the xitari reference", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
