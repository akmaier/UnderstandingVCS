"""Per-game scoring + termination rules.

The `RomSettings` base class is the contract front-ends use to read out
the current score, lives, and game-over flag from the running console.
Per-game subclasses inspect RAM at specific addresses (the addresses are
game-specific reverse-engineering work; see xitari/games/supported/* for
the original lookups).

P6 ships only `GenericRomSettings`, a no-op stub that reports zero
reward and never terminates. Real game settings (Pong, Breakout, Space
Invaders, Pitfall, etc.) land in a follow-up.
"""

from jaxtari.games.rom_settings import GenericRomSettings, RomSettings

__all__ = ["GenericRomSettings", "RomSettings"]
