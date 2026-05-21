"""StellaEnvironment — ALE-style RL interface over a `Console`.

This is the public API agents drive: `reset()` puts the console in a
fresh state with PC loaded from the cart's reset vector;
`step(action)` applies the action, runs the console until one frame
completes, and returns `(reward, terminal)`; `get_screen()` / `get_ram()`
expose the visible state.

Phosphor blending (Stella post-processes the framebuffer to smooth out
single-frame flicker) is intentionally absent in P6; it's a follow-up.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from jaxtari.console import Console, console_reset, initial_console, run_until_frame
from jaxtari.games.rom_settings import GenericRomSettings, RomSettings
from jaxtari.io.action import Action, apply_action


class StellaEnvironment:
    """A one-shot wrapper around a `Console` + `RomSettings`.

    Lifecycle:

      env = StellaEnvironment(rom_bytes)
      env.reset()
      while not env.game_over():
          reward = env.step(action)
          frame  = env.get_screen()
    """

    def __init__(self, rom, settings: Optional[RomSettings] = None) -> None:
        self._console: Console = initial_console(rom)
        self._settings: RomSettings = settings if settings is not None else GenericRomSettings()
        self._terminal: bool = False
        # `reset()` is required before `step` can be used; we don't call
        # it implicitly here so initial state is observable for tests.

    # --- Lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Reset the console (PC ← cart reset vector) and the settings.
        Steps a few frames so VSYNC has fired at least once and the
        framebuffer holds a meaningful image."""
        self._console = console_reset(self._console)
        self._settings.reset()
        self._terminal = False
        # Don't burn frames automatically — the caller decides whether
        # they want "noop skip" behaviour at the start of an episode.

    def step(self, action: int) -> int:
        """Apply `action`, run one console frame, return the per-step reward.

        After `step`, `game_over()` may transition to True and subsequent
        calls become no-ops returning 0.
        """
        if self._terminal:
            return 0
        self._console = apply_action(self._console, int(action))
        self._console = run_until_frame(self._console)
        reward = int(self._settings.get_reward(self._console))
        self._terminal = bool(self._settings.is_terminal(self._console))
        return reward

    # --- Observers ---------------------------------------------------------

    @property
    def console(self) -> Console:
        """Direct access to the underlying console — useful for tests and
        XAI work that needs to inspect register / RAM state."""
        return self._console

    def get_screen(self) -> jnp.ndarray:
        """Return the current framebuffer, shape (SCREEN_HEIGHT, SCREEN_WIDTH),
        uint8 indexed colour."""
        return self._console.bus.tia.framebuffer

    def get_ram(self) -> jnp.ndarray:
        """Return the 128-byte RIOT RAM."""
        return self._console.bus.ram

    def game_over(self) -> bool:
        return self._terminal

    def lives(self) -> int:
        return int(self._settings.lives(self._console))

    def frame_number(self) -> int:
        return int(self._console.bus.tia.frame)

    # --- ALE-API aliases ---------------------------------------------------

    # ALE traditionally names these in camelCase. Provide both spellings
    # so code written against the original ALE moves with minimal
    # changes.

    def act(self, action: int) -> int:
        return self.step(action)

    def getScreen(self) -> jnp.ndarray:
        return self.get_screen()

    def getRAM(self) -> jnp.ndarray:
        return self.get_ram()

    def getEpisodeFrameNumber(self) -> int:
        return self.frame_number()

    def gameOver(self) -> bool:
        return self.game_over()
