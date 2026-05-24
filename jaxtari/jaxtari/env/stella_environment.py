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

import random as _random
from typing import Optional

import jax.numpy as jnp

from jaxtari.console import Console, console_reset, initial_console, run_until_frame
from jaxtari.games.rom_settings import GenericRomSettings, RomSettings
from jaxtari.io.action import Action, apply_action, console_switches


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

    def reset(self, *, boot_noop_steps: int = 0,
              boot_reset_steps: int = 0,
              random_noop_max: int = 0,
              seed: Optional[int] = None) -> None:
        """Reset the console (PC ← cart reset vector) and the settings.

        Parameters
        ----------
        boot_noop_steps
            Number of NOOP frames to burn after the hardware reset
            before user actions start. **Default 0** — preserves the
            historical jaxtari behaviour where the caller decides the
            startup convention. Set to **60** for ALE / xitari parity
            (xitari's `resetGame()` burns 60 deterministic NOOP frames
            so the cart's startup routine has time to settle before
            "frame 1").
        boot_reset_steps
            Number of frames to burn with the console RESET switch held
            pressed, after the NOOP burn. **Default 0**. Set to **4**
            for ALE / xitari parity (xitari's `resetGame()` then holds
            the RESET switch for `system_reset_steps` frames, default 4).
        random_noop_max
            **P6d** — additional NOOP frames to burn at episode start,
            chosen uniformly from `[0, random_noop_max]`. **Default 0**
            (deterministic). Set to **30** for the canonical Mnih-style
            "skip 0..30 NOOPs at episode start" episode-randomization
            recipe — gives a stochastic-policy agent a different
            starting state per episode without affecting the
            deterministic xitari-parity startup. Sampled once per
            `reset()` call.
        seed
            Optional integer seed for the random-noop RNG. If `None`,
            uses the default `random` module state (so callers that
            seed `random.seed(...)` at the top of an experiment get
            reproducible runs across `reset()` calls).

        Together, `reset(boot_noop_steps=60, boot_reset_steps=4)`
        reproduces xitari's `ALEInterface::resetGame()` startup — the
        PXC1 conformance harness uses these values. Adding
        `random_noop_max=30` layers Mnih-style episode randomization
        on top.
        """
        if boot_noop_steps < 0 or boot_reset_steps < 0 or random_noop_max < 0:
            raise ValueError("boot_* / random_noop_max must be non-negative")
        self._console = console_reset(self._console)
        self._settings.reset()
        self._terminal = False

        # --- Boot-burn: NOOP frames -------------------------------------- #
        for _ in range(boot_noop_steps):
            self._console = apply_action(self._console, int(Action.NOOP))
            self._console = run_until_frame(self._console)

        # --- Boot-burn: RESET-switch frames ------------------------------ #
        if boot_reset_steps > 0:
            self._console = console_switches(self._console, reset_pressed=True)
            for _ in range(boot_reset_steps):
                self._console = apply_action(self._console, int(Action.NOOP))
                self._console = run_until_frame(self._console)
            self._console = console_switches(self._console, reset_pressed=False)

        # --- P6d: random-NOOP episode randomization ---------------------- #
        if random_noop_max > 0:
            rng = _random.Random(seed) if seed is not None else _random
            n = rng.randint(0, random_noop_max)
            for _ in range(n):
                self._console = apply_action(self._console, int(Action.NOOP))
                self._console = run_until_frame(self._console)

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
