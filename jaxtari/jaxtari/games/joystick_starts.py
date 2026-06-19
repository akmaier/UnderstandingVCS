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

from jaxtari.console import Console
from jaxtari.games.rom_settings import GenericRomSettings


class WizardOfWorRomSettings(GenericRomSettings):
    # wizard_of_wor's single-player agent is the RIGHT player (P1, SWCHA low
    # nibble) in xitari/ALE, not the default LEFT (P0). Verified by the SWCHA
    # read on the first in-play joystick press (agent RIGHT clears bit 3, not
    # bit 7). Routing to P0 was bit-exact through the conformance window but
    # diverged the instant gameplay reads the stick (~frame 217). No starting
    # actions. Mirror of jutari WizardOfWorRomSettings.
    def agent_player(self) -> int:
        return 1


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


# Asterix terminal addresses — #127b sprint 3 (xitari Asterix.cpp::step):
#   m_lives       = readRam(0xD3) & 0xF
#   death_counter = readRam(0xC7)
#   terminal = (death_counter == 0x01 && m_lives == 1)
# (xitari can't wait for lives==0 because the agent may restart on the last
# frame by holding fire.) The long-horizon run dies at action-frame 1158;
# xitari auto-resets (lives→3) while the render-only Asterix settings (no
# terminal reader → false) kept rendering the dead episode — the f1160 "TIA
# pixel diff" #127b MISCLASSIFIED as Cluster A render. This extends the existing
# FIRE-start class so Asterix keeps starting_actions=[FIRE].
ASTERIX_LIVES_ADDR = 0xD3
ASTERIX_DEATH_ADDR = 0xC7


class AsterixRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE

    def is_terminal(self, console: Console) -> bool:
        ram = console.bus.ram
        lives = int(ram[ASTERIX_LIVES_ADDR & 0x7F]) & 0xF
        death_counter = int(ram[ASTERIX_DEATH_ADDR & 0x7F])
        return death_counter == 0x01 and lives == 1


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


# Pooyan terminal addresses — #127b sprint 4 (xitari Pooyan.cpp::step):
#   lives_byte = readRam(0x96)
#   some_byte  = readRam(0x98)
#   terminal = (lives_byte == 0x0 && some_byte == 0x05)
# #127b first flagged pooyan as "the one genuine TIA render bug" (f1605); that
# was a TOOLING ARTIFACT (the probes hardcode H=210, pooyan renders at H=220 →
# fake localized diff). At H=220 the real first-div is f1532 — pooyan's
# game-over frame; a terminal/auto-reset gap like the rest. Boot RAM[0x96]=2 →
# terminal false through the in-window sweep. Extends the existing render-only
# class so Pooyan keeps height=220 / y_start=26.
POOYAN_LIVES_ADDR = 0x96
POOYAN_STATE_ADDR = 0x98
POOYAN_DEATH_STATE = 0x05


class PooyanRomSettings(GenericRomSettings):
    # Task #110 follow-up: stella.pro Display.YStart=26 / Display.Height=220.
    # NTSC, render-only crop overrides (same shape as Carnival).
    def screen_height(self) -> int:
        return 220

    def screen_y_start(self) -> int:
        return 26

    def is_terminal(self, console: Console) -> bool:
        ram = console.bus.ram
        lives_byte = int(ram[POOYAN_LIVES_ADDR & 0x7F])
        some_byte = int(ram[POOYAN_STATE_ADDR & 0x7F])
        return lives_byte == 0x0 and some_byte == POOYAN_DEATH_STATE


class BattleZoneRomSettings(GenericRomSettings):
    # Task #111: battle_zone's stella.pro entry sets
    # `Emulation.HmoveBlanks "NO"`. battle_zone strobes HMOVE every visible
    # scanline (cc 222), and without this disable the comb would blank cols
    # 0-7 on every row (1112 px). Render-only; no starting actions.
    def hmove_blanks(self) -> bool:
        return False


# NB: MsPacmanRomSettings lives in `more_games.py` (it carries the RL
# score/lives/terminal decoding AND `hmove_blanks() == False`). A
# hmove_blanks-only stub used to live here too and SHADOWED the real one in
# the games package export, silently dropping the decoding (task #125). The
# stub was removed; import MsPacmanRomSettings from `more_games`.


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


# Pac-Man terminal addresses — #127b sprint 4 (xitari Pacman.cpp::step):
#   m_lives          = readRam(0x98) + 1     (so RAM[0x98]==0 ⇔ m_lives==1)
#   animationCounter = readRam(0xE4)
#   terminal = (m_lives == 1 && animationCounter == 0x3F)
# Pac-Man (distinct from ms_pacman). The long-horizon run dies at action-frame
# 1770 (RAM[0x98]→0, RAM[0xE4]→0x3F); xitari auto-resets (lives→4) while the
# render-only Pacman settings (no terminal reader → false) kept rendering the
# dead episode — the f1771 tail #127b MISCLASSIFIED as Cluster A render. Boot
# RAM[0x98]=3 → terminal false. Extends the existing render-only class so Pacman
# keeps y_start=33.
PACMAN_LIVES_ADDR = 0x98
PACMAN_ANIM_ADDR  = 0xE4
PACMAN_DEATH_ANIM = 0x3F


class PacmanRomSettings(GenericRomSettings):
    # Task #113: pacman (NOT ms_pacman) has Display.YStart=33 in stella.pro.
    # Render-only crop override; no starting actions.
    def screen_y_start(self) -> int:
        return 33

    def is_terminal(self, console: Console) -> bool:
        ram = console.bus.ram
        lives_byte = int(ram[PACMAN_LIVES_ADDR & 0x7F])
        animation_counter = int(ram[PACMAN_ANIM_ADDR & 0x7F])
        return lives_byte == 0 and animation_counter == PACMAN_DEATH_ANIM


# Phoenix terminal address — #127b sprint 4 (xitari Phoenix.cpp::step):
#   state_byte = readRam(0xCC)
#   terminal = (state_byte == 0x80)
#   m_lives  = readRam(0xCB) & 0x7   (starts at 5)
# Phoenix has no stella.pro render override (default NTSC) and no
# getStartingActions, so this class behaves like GenericRomSettings except for
# the terminal reader. The long-horizon run dies at action-frame 1742
# (RAM[0xCC]→0x80); xitari auto-resets (lives→5) while a settings-less front-end
# (Generic → never terminal) kept rendering the dead episode — the f1743
# whole-screen swap #127b flagged "Cluster A render (medium confidence)" is this
# terminal/auto-reset gap. Boot RAM[0xCC]=0 → terminal false. New class.
PHOENIX_STATE_ADDR = 0xCC
PHOENIX_DEATH_STATE = 0x80
PHOENIX_LIVES_ADDR = 0xCB


class PhoenixRomSettings(GenericRomSettings):
    def is_terminal(self, console: Console) -> bool:
        return int(console.bus.ram[PHOENIX_STATE_ADDR & 0x7F]) == PHOENIX_DEATH_STATE

    def lives(self, console: Console) -> int:
        return int(console.bus.ram[PHOENIX_LIVES_ADDR & 0x7F]) & 0x7


class YarsRevengeRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE
