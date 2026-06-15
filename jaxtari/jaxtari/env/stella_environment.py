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
from jaxtari.tia.system import set_paddle_resistance


# Task #54 — paddle-action support. xitari's `ALEState::applyActionPaddles`
# (xitari/environment/ale_state.cpp:150) converts LEFT/RIGHT actions
# into paddle-position deltas in xitari's resistance scale (PADDLE_MIN
# = 27 450, PADDLE_MAX = 790 196, PADDLE_DELTA = 23 000 per frame).
# Default position is the midpoint. We use the *same* numbers so the
# paddle motion produced by a given action sequence matches xitari's.
_PADDLE_MIN     = 27_450
_PADDLE_MAX     = 790_196
_PADDLE_DELTA   = 23_000
_PADDLE_DEFAULT = (_PADDLE_MAX - _PADDLE_MIN) // 2 + _PADDLE_MIN

# Action sets that move the LEFT or RIGHT paddle in xitari's
# `applyActionPaddles`. The semantic is "LEFT-direction action moves
# the paddle in the +delta direction" — i.e. LEFT and LEFTFIRE
# (and all *LEFT* compound actions) push the left paddle's
# resistance UP; the same for RIGHT* and the right paddle.
_ACTIONS_LEFT_PADDLE_INC  = {Action.LEFT,  Action.LEFTFIRE,
                             Action.UPLEFT,  Action.DOWNLEFT,
                             Action.UPLEFTFIRE,  Action.DOWNLEFTFIRE}
_ACTIONS_LEFT_PADDLE_DEC  = {Action.RIGHT, Action.RIGHTFIRE,
                             Action.UPRIGHT, Action.DOWNRIGHT,
                             Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE}


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
        """Construct an env over `rom`. `settings` is the per-game
        RomSettings (defaults to GenericRomSettings).

        Whether LEFT/RIGHT actions drive a joystick or a paddle is
        decided by `settings.uses_paddles()` — auto-detected per game
        from the settings class (BreakoutRomSettings / PongRomSettings
        / …). xitari handles the same decision from stella.pro's
        `Controller.Left "PADDLES"` property; jaxtari encodes it on
        the per-game settings class so the env stays oblivious to
        stella.pro.
        """
        self._console: Console = initial_console(rom)
        self._settings: RomSettings = settings if settings is not None else GenericRomSettings()
        self._terminal: bool = False
        # Default both paddles to centre. Written to the TIA on every
        # step when `settings.uses_paddles()` is True. Stays unused
        # otherwise.
        self._left_paddle:  int = _PADDLE_DEFAULT
        self._right_paddle: int = _PADDLE_DEFAULT
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
        # PXC1-x round 5: for paddle games, push the default paddle
        # resistance into the TIA BEFORE the boot-burn loop runs.
        # xitari constructs `StellaEnvironment` with `resetPaddles` —
        # which sets PaddleZeroResistance / PaddleOneResistance via
        # the Event mechanism — BEFORE the 60-NOOP boot frames run, so
        # any INPT0/INPT1 reads the ROM does during boot see the
        # dump-pot capacitor timing path from the start. Without this
        # pre-boot apply, jaxtari's boot frames see the static $80
        # default for INPT0/INPT1 (the `inpt` array), which diverges
        # from xitari's cycle-threshold dump-pot result by the time
        # the 60-NOOP/4-RESET burn finishes. With the pre-boot apply,
        # both paddles are at PADDLE_DEFAULT in dump-pot mode for the
        # whole boot phase.
        self._left_paddle  = _PADDLE_DEFAULT
        self._right_paddle = _PADDLE_DEFAULT
        if self._settings.uses_paddles():
            self._apply_paddle_action(int(Action.NOOP))

        # --- Console difficulty switches (#103, amidar) ------------------ #
        # xitari's default properties are B/B (SWCHB 0x3F); a ROM's
        # stella.pro entry can override a difficulty to "A" (sets the SWCHB
        # bit → 0xFF for A/A). amidar = A/A and reads the P0/Left bit in its
        # frame-1 object sort, so the difficulty must be correct from the
        # FIRST boot frame — apply here (before the NOOP boot) and re-assert
        # in every console_switches call below (each rebuilds the whole
        # SWCHB byte). Default (False, False) = B/B, unchanged for the
        # bit-exact games. Defensive getattr: older RomSettings lack it.
        try:
            diff = self._settings.difficulty()
            diff0, diff1 = ((bool(diff[0]), bool(diff[1]))
                            if diff is not None else (False, False))
        except AttributeError:
            diff0, diff1 = False, False
        self._console = console_switches(
            self._console, p0_difficulty_a=diff0, p1_difficulty_a=diff1)

        # --- Boot-burn: NOOP frames -------------------------------------- #
        for _ in range(boot_noop_steps):
            self._console = apply_action(self._console, int(Action.NOOP))
            self._console = run_until_frame(self._console)

        # --- Boot-burn: RESET-switch frames ------------------------------ #
        if boot_reset_steps > 0:
            self._console = console_switches(
                self._console, reset_pressed=True,
                p0_difficulty_a=diff0, p1_difficulty_a=diff1)
            for _ in range(boot_reset_steps):
                self._console = apply_action(self._console, int(Action.NOOP))
                self._console = run_until_frame(self._console)
            self._console = console_switches(
                self._console, reset_pressed=False,
                p0_difficulty_a=diff0, p1_difficulty_a=diff1)

        # --- Per-game starting actions ----------------------------------- #
        # xitari's `StellaEnvironment::reset` emulates
        # `m_settings->getStartingActions()` AFTER the boot burn (and
        # AFTER `m_settings->reset()`, which we did above on line 132).
        # Pitfall: 1× PLAYER_A_UP. Enduro: 1× PLAYER_A_FIRE.
        # Default `use_starting_actions = true` in
        # xitari/common/Defaults.cpp. Without this we're 1 frame behind
        # xitari for those ROMs — the documented 19/45 b/f RAM
        # divergence at frame 0 (tasks #81/#82).
        try:
            sa = self._settings.starting_actions()
            # RomSettings is a Protocol — subclasses that don't override
            # `starting_actions` inherit the abstract version which
            # returns `None`. Treat that as "no starting actions" rather
            # than crashing on `list(None)`.
            starting = list(sa) if sa is not None else []
        except AttributeError:
            # Older RomSettings subclasses (pre-task-#81 ports) may not
            # implement starting_actions yet. Treat as empty.
            starting = []
        uses_paddles_sa = self._settings.uses_paddles()
        swap_paddles_sa = self._settings.swap_paddles() if uses_paddles_sa else False
        for action in starting:
            if uses_paddles_sa:
                self._apply_paddle_action(int(action))
            self._console = apply_action(
                self._console, int(action),
                paddle_mode=uses_paddles_sa, swap_paddles=swap_paddles_sa)
            self._console = run_until_frame(self._console)

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

        Task #54 — when `use_paddles=True` was passed to `__init__`,
        LEFT/RIGHT actions also nudge the left paddle's resistance by
        `±PADDLE_DELTA` (xitari's `applyActionPaddles` semantic) and
        write the new value to the TIA via `set_paddle_resistance` so
        Breakout / Pong / Warlords / ... see the paddle move.
        """
        if self._terminal:
            return 0
        uses_paddles = self._settings.uses_paddles()
        if uses_paddles:
            self._apply_paddle_action(int(action))
        # Task #65: when paddles are in play, xitari's `applyActionPaddles`
        # only sets paddle resistance + fire event — NOT the joystick
        # direction (SWCHA). Forwarding LEFT/RIGHT to SWCHA on a paddle
        # game like pong made the game branch differently from xitari
        # (paddle stuck at default position even though paddle_resistance
        # was being updated correctly). `paddle_mode=True` skips the
        # SWCHA write and just updates the trigger from the action's
        # fire bit, matching xitari.
        # Task #77 (2026-06-07): with SwapPaddles=YES (Pong) the USER's
        # FIRE button needs to route to paddle 1's SWCHA bit (= bit 6,
        # P0_LEFT) instead of paddle 0's (= bit 7, P0_RIGHT). Thread
        # the swap flag through so apply_action can swap accordingly.
        swap = self._settings.swap_paddles() if uses_paddles else False
        self._console = apply_action(self._console, int(action),
                                      paddle_mode=uses_paddles,
                                      swap_paddles=swap)
        self._console = run_until_frame(self._console)
        reward = int(self._settings.get_reward(self._console))
        self._terminal = bool(self._settings.is_terminal(self._console))
        return reward

    def _apply_paddle_action(self, action: int) -> None:
        """xitari `applyActionPaddles` — translate an action into
        ±PADDLE_DELTA on the left paddle (the right paddle stays put
        because the action enum only encodes one player). The new
        position is converted to a paddle resistance and written into
        the TIA so INPT0's dump-pot cycle threshold reflects the
        paddle position."""
        if action in _ACTIONS_LEFT_PADDLE_INC:
            self._left_paddle += _PADDLE_DELTA
        elif action in _ACTIONS_LEFT_PADDLE_DEC:
            self._left_paddle -= _PADDLE_DELTA
        # Clamp.
        if self._left_paddle < _PADDLE_MIN:
            self._left_paddle = _PADDLE_MIN
        elif self._left_paddle > _PADDLE_MAX:
            self._left_paddle = _PADDLE_MAX
        # Task #73 (2026-06-07): route paddle resistance per xitari Paddles
        # wiring. With SwapPaddles=NO (Breakout convention) the user paddle
        # reaches INPT0 (Pin Nine on the left controller); with
        # SwapPaddles=YES (Pong / Video Olympics) it reaches INPT1
        # (Pin Five). xitari handles this inside
        # `Paddles::Paddles(jack, event, swap)` via the
        # `myPinEvents[2/3][0/1]` wiring table — see
        # `xitari/emucore/Paddles.cxx`. Without this routing fix, jaxtari's
        # pong paddle update lands on INPT0 which the game never reads, so
        # the on-screen paddle stays frozen at the centred default
        # (RAM $04 / $3c never decrement past the initial 0x6e / 0x6d).
        # Mirror of jutari task #66 / commit 8531bb8.
        if self._settings.swap_paddles():
            new_tia = set_paddle_resistance(self._console.bus.tia, 1, self._left_paddle)
            new_tia = set_paddle_resistance(new_tia,                0, self._right_paddle)
        else:
            new_tia = set_paddle_resistance(self._console.bus.tia, 0, self._left_paddle)
            new_tia = set_paddle_resistance(new_tia,                1, self._right_paddle)
        new_bus = self._console.bus._replace(tia=new_tia)
        self._console = self._console._replace(bus=new_bus)

    # --- Observers ---------------------------------------------------------

    @property
    def console(self) -> Console:
        """Direct access to the underlying console — useful for tests and
        XAI work that needs to inspect register / RAM state."""
        return self._console

    def get_screen(self) -> jnp.ndarray:
        """Return the visible portion of the current framebuffer.

        Shape `(VISIBLE_HEIGHT, SCREEN_WIDTH) = (210, 160)`, uint8 indexed
        colour — matches xitari/ALE's `Display.YStart=34`/`Display.Height=210`
        default crop. The top 34 scanlines (VSYNC + VBLANK + any
        score-header area outside xitari's display window) are cropped
        out, so jaxtari videos line up vertically with xitari videos.

        The full internal framebuffer (244 rows, scanlines 0..243) is
        still on `console.bus.tia.framebuffer` for tests / debugging
        that want the uncropped view.
        """
        from jaxtari.tia.system import Y_START, VISIBLE_HEIGHT
        fb = self._console.bus.tia.framebuffer
        return fb[Y_START : Y_START + VISIBLE_HEIGHT]

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
