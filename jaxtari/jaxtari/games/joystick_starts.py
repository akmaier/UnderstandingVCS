"""Task #101 — minimal RomSettings for joystick games whose only
conformance-relevant per-game behavior is xitari's `getStartingActions()`
(a single action the ALE harness applies AFTER the boot burn). Mirror of
jutari `src/games/JoystickGames.jl`.

Without applying the starting action our generic NOOP boot is one action
behind xitari → frame-0 RAM divergence in the 64-ROM sweep. These subclass
`GenericRomSettings` (no-op scoring) and override only `starting_actions`.
Action codes per xitari `ale_interface.hpp`: FIRE=1, UP=2, RIGHT=3, DOWN=5,
UPFIRE=10. Source: each game's `xitari/games/supported/<Game>.cpp::
getStartingActions`. (beam_rider's start action [RIGHT]=3 lives on the
existing `BeamriderRomSettings` scorer in atari_classics.py.)
"""

from __future__ import annotations

from jaxtari.games.rom_settings import GenericRomSettings


class AmidarRomSettings(GenericRomSettings):
    # Task #103: amidar's stella.pro overrides BOTH console difficulty
    # switches to "A" → SWCHB 0xFF (xitari default is B/B = 0x3F). amidar
    # reads the P0/Left difficulty bit during its frame-1 object sort, so
    # without A/A it sorts the wrong way (11 b/f). No starting actions.
    def difficulty(self) -> tuple[bool, bool]:
        return (True, True)   # A/A


class AirRaidRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE

    def pal(self) -> bool:
        # air_raid is a PAL dump (its real VSYNC at scanline 286 ends the
        # frame before either 290 or 342, so the threshold doesn't change its
        # result — flagged for correctness / xitari parity). See task #103.
        return True

    def screen_height(self) -> int:
        return 250   # task #110 (PAL bump 210→250)


class AsterixRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE


class DoubleDunkRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [10]  # UPFIRE


class ElevatorActionRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1] * 16   # 16× FIRE (xitari ElevatorAction.cpp loop)


class GopherRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE


class GravitarRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1] * 16   # 16× FIRE (xitari Gravitar.cpp loop)


class JourneyEscapeRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE

    def screen_height(self) -> int:
        return 230   # task #110 (stella.pro Display.Height)


class CarnivalRomSettings(GenericRomSettings):
    # Task #110 follow-up: stella.pro Display.YStart=26 / Display.Height=214.
    # NTSC (rendered content stays within scanline 262), so no PAL flag —
    # render-only crop overrides.
    def screen_height(self) -> int:
        return 214

    def screen_y_start(self) -> int:
        return 26


class PooyanRomSettings(GenericRomSettings):
    # Task #110 follow-up: stella.pro Display.YStart=26 / Display.Height=220.
    # NTSC, render-only crop overrides (same shape as Carnival).
    def screen_height(self) -> int:
        return 220

    def screen_y_start(self) -> int:
        return 26


class BattleZoneRomSettings(GenericRomSettings):
    # Task #111: battle_zone's stella.pro entry sets
    # `Emulation.HmoveBlanks "NO"`. battle_zone strobes HMOVE every visible
    # scanline (cc 222), and without this disable the comb would blank cols
    # 0-7 on every row (1112 px). Render-only; no starting actions.
    def hmove_blanks(self) -> bool:
        return False


class MsPacmanRomSettings(GenericRomSettings):
    # Task #111: ms_pacman's stella.pro entry sets
    # `Emulation.HmoveBlanks "NO"`. Same fix as battle_zone — disable the
    # HMOVE comb. Render-only; no starting actions.
    def hmove_blanks(self) -> bool:
        return False


class PrivateEyeRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [2]   # UP


class SkiingRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [5] * 16   # 16× DOWN (xitari Skiing.cpp loop)

    def is_legal_action(self, action: int) -> bool:
        # xitari SkiingSettings::isLegal (Skiing.cpp:96-111) rejects the whole
        # FIRE family; noopIllegalActions maps them to NOOP before a user step.
        # Skiing is the only supported game overriding isLegal. Without this,
        # the sweep's shared breakout stream injects FIRE at frame 20 and
        # skiing diverges 84 b/f. FIRE-family ALE codes. See task #103.
        return action not in (1, 10, 11, 12, 13, 14, 15, 16, 17)


class SurroundRomSettings(GenericRomSettings):
    # Task #103: surround is a PAL game; xitari getStartingActions =
    # {SELECT, RESET} (Surround.cpp:135) selects game variation 1 then starts
    # it. SELECT/RESET are console switches (not joystick), routed via
    # console_switches in env.reset(). PAL → 342 max-scanlines cutoff (its
    # 312-line frame would otherwise be split by the NTSC 290 cutoff).
    def console_switch_starts(self) -> list[int]:
        return [46, 40]   # SELECT, RESET

    def pal(self) -> bool:
        return True

    def screen_height(self) -> int:
        return 250   # task #110 (PAL bump 210→250)


class UpNDownRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE

    def screen_y_start(self) -> int:
        return 30   # task #113 (stella.pro Display.YStart)


class PacmanRomSettings(GenericRomSettings):
    # Task #113: pacman (NOT ms_pacman) has Display.YStart=33 in stella.pro.
    # Render-only crop override; no starting actions.
    def screen_y_start(self) -> int:
        return 33


class YarsRevengeRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE
