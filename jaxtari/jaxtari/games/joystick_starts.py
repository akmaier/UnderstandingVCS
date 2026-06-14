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


class AirRaidRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE


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


class PrivateEyeRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [2]   # UP


class SkiingRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [5] * 16   # 16× DOWN (xitari Skiing.cpp loop)


class UpNDownRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE


class YarsRevengeRomSettings(GenericRomSettings):
    def starting_actions(self) -> list[int]:
        return [1]   # FIRE
