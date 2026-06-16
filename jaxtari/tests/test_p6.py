"""P6 tests — Console wiring, IO actions, StellaEnvironment lifecycle."""

import jax.numpy as jnp
import pytest

from jaxtari.console import (
    Console,
    console_reset,
    console_step,
    initial_console,
    run_until_frame,
)
from jaxtari.env import StellaEnvironment
from jaxtari.io.action import Action, apply_action, console_switches
from jaxtari.tia.system import set_trigger


# --------------------------------------------------------------------------- #
# Test ROMs
# --------------------------------------------------------------------------- #

def _frame_loop_rom() -> jnp.ndarray:
    """Minimal 4K ROM that endlessly pulses VSYNC. Each loop iteration
    flips VSYNC.D1 on then off → one frame boundary per iteration."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    program = [
        0xA9, 0x02,        # LDA #$02
        0x85, 0x00,        # STA VSYNC
        0xA9, 0x00,        # LDA #$00
        0x85, 0x00,        # STA VSYNC  → falling edge → frame++
        0x4C, 0x00, 0xF0,  # JMP $F000
    ]
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.uint8(b))
    # Reset vector at $FFFC/$FFFD → ROM offsets $0FFC/$0FFD for a 4K cart.
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    return rom


def _ram_reader_rom() -> jnp.ndarray:
    """ROM that reads SWCHA into RAM each frame, so a test can verify
    that apply_action set the joystick bits correctly."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    program = [
        0xAD, 0x80, 0x02,  # LDA $0280  — SWCHA
        0x85, 0x80,        # STA $80    — RAM[0] (canonical)
        0xAD, 0x82, 0x02,  # LDA $0282  — SWCHB
        0x85, 0x81,        # STA $81
        0xA9, 0x02, 0x85, 0x00,  # LDA #$02 / STA VSYNC
        0xA9, 0x00, 0x85, 0x00,  # LDA #$00 / STA VSYNC  → frame++
        0x4C, 0x00, 0xF0,  # JMP $F000
    ]
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.uint8(b))
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    return rom


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #

def test_initial_console_default_pc_is_zero():
    """Without `reset`, PC is whatever `initial_cpu_state()` set it to (0)."""
    c = initial_console(_frame_loop_rom())
    assert int(c.cpu.PC) == 0


def test_console_reset_loads_pc_from_reset_vector():
    c = initial_console(_frame_loop_rom())
    c = console_reset(c)
    assert int(c.cpu.PC) == 0xF000


def test_console_reset_zeroes_ram_and_tia():
    """A fresh console after reset has zero RAM, zero TIA frame counter,
    and a zero framebuffer."""
    c = initial_console(_frame_loop_rom())
    # Dirty things first.
    c2 = c._replace(bus=c.bus._replace(
        ram=c.bus.ram.at[0].set(jnp.uint8(0xAA)),
    ))
    c2 = c2._replace(bus=c2.bus._replace(tia=c2.bus.tia._replace(frame=42)))
    c3 = console_reset(c2)
    assert int(c3.bus.ram[0]) == 0
    assert int(c3.bus.tia.frame) == 0
    assert int(c3.bus.tia.framebuffer.sum()) == 0


def test_console_step_advances_one_instruction():
    c = console_reset(initial_console(_frame_loop_rom()))
    pc_before = int(c.cpu.PC)
    c = console_step(c)
    # First instruction was LDA #$02 (2 bytes).
    assert int(c.cpu.PC) == pc_before + 2


def test_run_until_frame_returns_after_one_frame():
    c = console_reset(initial_console(_frame_loop_rom()))
    assert int(c.bus.tia.frame) == 0
    c = run_until_frame(c)
    assert int(c.bus.tia.frame) == 1


def test_run_until_frame_advances_two_frames_back_to_back():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = run_until_frame(c)
    c = run_until_frame(c)
    assert int(c.bus.tia.frame) == 2


def test_run_until_frame_greys_poke_less_rom():
    """Task #106 (partial-frame model): xitari's max-scanlines cutoff
    (TIA.cxx:2003) is evaluated at POKE time only, NOT every CPU step. So a
    poke-LESS, VSYNC-less ROM never ends a frame — it runs "grey" and
    `run_until_frame` returns at the 25000-instruction `execute(25000)`
    budget WITHOUT advancing the frame counter (xitari leaves
    myPartialFrameFlag true / getFrameNumber unchanged). It must not hang
    or raise."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0x4C))      # JMP $F000 (no pokes)
    rom = rom.at[1].set(jnp.uint8(0x00))
    rom = rom.at[2].set(jnp.uint8(0xF0))
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    c = console_reset(initial_console(rom))
    f0 = int(c.bus.tia.frame)
    c = run_until_frame(c)
    assert int(c.bus.tia.frame) == f0                 # grey frame: NO advance
    assert int(c.bus.tia.lines_since_frame) > 290     # beam ran past the cutoff


def test_run_until_frame_poke_time_cutoff_ends_vsyncless_rom():
    """...but a VSYNC-less ROM that DOES poke a TIA register every loop ends
    its frame at the next poke past 290 scanlines (the real seaquest/air_raid
    boot-burst mechanism, tasks #80/#108; xitari TIA.cxx:2003-2007)."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0x8D))      # STA $0009 (COLUBK)
    rom = rom.at[1].set(jnp.uint8(0x09))
    rom = rom.at[2].set(jnp.uint8(0x00))
    rom = rom.at[3].set(jnp.uint8(0x4C))      # JMP $F000
    rom = rom.at[4].set(jnp.uint8(0x00))
    rom = rom.at[5].set(jnp.uint8(0xF0))
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    c = console_reset(initial_console(rom))
    g0 = int(c.bus.tia.frame)
    c = run_until_frame(c)
    assert int(c.bus.tia.frame) == g0 + 1             # poke-time cutoff fired
    assert int(c.bus.tia.lines_since_frame) < 290      # new frame started fresh


# --------------------------------------------------------------------------- #
# IO actions — joystick decoding
# --------------------------------------------------------------------------- #

def _swcha_after(action: int) -> int:
    """Apply `action` to a fresh console and return the resulting SWCHA
    input byte."""
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, action)
    return int(c.bus.riot.swcha_in)


def test_action_noop_all_directions_released():
    # SWCHA all high = no direction asserted. P1 nibble = 0x0F.
    assert _swcha_after(Action.NOOP) == 0xFF


def test_action_up_clears_p0_up_bit():
    # P0 UP = bit 4; pressed = 0. Others stay high.
    assert _swcha_after(Action.UP) == 0xEF


def test_action_right_clears_p0_right_bit():
    assert _swcha_after(Action.RIGHT) == 0x7F


def test_action_left_clears_p0_left_bit():
    assert _swcha_after(Action.LEFT) == 0xBF


def test_action_down_clears_p0_down_bit():
    assert _swcha_after(Action.DOWN) == 0xDF


def test_action_upright_clears_both_bits():
    # UP (bit 4) and RIGHT (bit 7) cleared simultaneously.
    assert _swcha_after(Action.UPRIGHT) == 0x6F


def test_action_downleft_clears_both_bits():
    # DOWN (bit 5) and LEFT (bit 6) cleared simultaneously.
    assert _swcha_after(Action.DOWNLEFT) == 0x9F


# --------------------------------------------------------------------------- #
# IO actions — fire button
# --------------------------------------------------------------------------- #

def test_action_fire_sets_inpt4_pressed():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.FIRE)
    assert int(c.bus.tia.inpt[4]) == 0x00      # active-low: pressed = 0


def test_action_noop_leaves_inpt4_released():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.NOOP)
    assert int(c.bus.tia.inpt[4]) == 0x80


def test_action_upfire_combines_direction_and_fire():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.UPFIRE)
    assert int(c.bus.riot.swcha_in) == 0xEF
    assert int(c.bus.tia.inpt[4]) == 0x00


# --------------------------------------------------------------------------- #
# IO — console switches
# --------------------------------------------------------------------------- #

def test_default_swchb_matches_xitari():
    """Task #64: `console_switches()` defaults now match xitari's
    `Switches::Switches` initial state — B/B difficulty (bits 6+7
    cleared) + COLOR (bit 3 set) + SELECT/RESET released = 0x3F.
    For Atari Breakout, bit 7 doubles as the paddle-size toggle:
    A=1 draws a 4-bit small paddle, B=0 draws an 8-bit large
    paddle. The old A/A default was the reason our Breakout
    paddle rendered smaller than xitari's."""
    c = console_reset(initial_console(_frame_loop_rom()))
    c = console_switches(c)                    # all defaults — now B/B
    assert int(c.bus.riot.swchb_in) == 0x3F


def test_console_switches_select_and_reset():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = console_switches(c, select_pressed=True, reset_pressed=True)
    # B/B difficulty default (0x3F) with SELECT (bit 1) + RESET (bit 0)
    # both cleared → 0x3C.
    assert int(c.bus.riot.swchb_in) == 0x3F & ~0x03


def test_console_switches_bw_mode():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = console_switches(c, color=False)
    assert int(c.bus.riot.swchb_in) & 0x08 == 0


def test_console_switches_difficulty_b():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = console_switches(c, p0_difficulty_a=False, p1_difficulty_a=False)
    assert int(c.bus.riot.swchb_in) & 0xC0 == 0


# --------------------------------------------------------------------------- #
# End-to-end via the ROM that reads SWCHA into RAM
# --------------------------------------------------------------------------- #

def test_action_visible_to_rom_via_ram():
    """The ROM reads SWCHA each frame and stashes it in RAM[$80]. After
    pressing UP, the next frame should see RAM[$80] = 0xEF."""
    c = console_reset(initial_console(_ram_reader_rom()))
    c = apply_action(c, Action.UP)
    c = run_until_frame(c)
    assert int(c.bus.ram[0]) == 0xEF


# --------------------------------------------------------------------------- #
# P6e — two-player joystick routing (player=1 driving SWCHA low nibble)
# --------------------------------------------------------------------------- #

def test_action_p1_noop_leaves_swcha_idle():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.NOOP, player=1)
    assert int(c.bus.riot.swcha_in) == 0xFF


def test_action_p1_up_clears_low_nibble_bit_0():
    """P1 UP = SWCHA bit 0 (active low). All other bits stay high."""
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.UP, player=1)
    assert int(c.bus.riot.swcha_in) == 0xFE


def test_action_p1_right_clears_bit_3():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.RIGHT, player=1)
    assert int(c.bus.riot.swcha_in) == 0xF7


def test_action_p1_left_clears_bit_2():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.LEFT, player=1)
    assert int(c.bus.riot.swcha_in) == 0xFB


def test_action_p1_down_clears_bit_1():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.DOWN, player=1)
    assert int(c.bus.riot.swcha_in) == 0xFD


def test_action_p1_upright_clears_bits_0_and_3():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.UPRIGHT, player=1)
    assert int(c.bus.riot.swcha_in) == 0xF6


def test_action_p1_fire_sets_inpt5_pressed():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.FIRE, player=1)
    assert int(c.bus.tia.inpt[5]) == 0x00     # P1 trigger lives in INPT5
    assert int(c.bus.tia.inpt[4]) == 0x80     # P0 trigger untouched


def test_p0_and_p1_actions_compose():
    """Two consecutive apply_action calls — one per player — must end up
    with both players' bits set in SWCHA simultaneously."""
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.UP,    player=0)   # clears bit 4 → 0xEF
    c = apply_action(c, Action.RIGHT, player=1)   # clears bit 3 → 0xE7
    assert int(c.bus.riot.swcha_in) == 0xE7
    # Triggers: neither was FIRE, so both stay released.
    assert int(c.bus.tia.inpt[4]) == 0x80
    assert int(c.bus.tia.inpt[5]) == 0x80


def test_p1_action_does_not_clobber_p0_nibble():
    """Driving P1 must leave the P0 nibble at whatever the previous
    apply_action for P0 set it to — the two players' inputs must be
    independent."""
    c = console_reset(initial_console(_frame_loop_rom()))
    c = apply_action(c, Action.DOWNLEFT, player=0)   # 0x9F
    c = apply_action(c, Action.DOWN,     player=1)   # clears bit 1 → 0x9D
    assert int(c.bus.riot.swcha_in) == 0x9D


def test_apply_action_rejects_invalid_player():
    c = console_reset(initial_console(_frame_loop_rom()))
    with pytest.raises(ValueError):
        apply_action(c, Action.NOOP, player=2)


# --------------------------------------------------------------------------- #
# StellaEnvironment
# --------------------------------------------------------------------------- #

def test_env_constructs_with_rom():
    env = StellaEnvironment(_frame_loop_rom())
    assert isinstance(env.console, Console)


def test_env_reset_then_step_returns_zero_reward_with_generic_settings():
    env = StellaEnvironment(_frame_loop_rom())
    env.reset()
    reward = env.step(Action.NOOP)
    assert reward == 0
    assert env.game_over() is False


def test_env_step_advances_frame_counter():
    env = StellaEnvironment(_frame_loop_rom())
    env.reset()
    assert env.frame_number() == 0
    env.step(Action.NOOP)
    assert env.frame_number() == 1
    env.step(Action.NOOP)
    assert env.frame_number() == 2


def test_env_get_screen_returns_framebuffer_shape():
    """`env.get_screen()` returns the ALE/xitari visible region —
    `(screen_height_rows=210, SCREEN_WIDTH=160)` for default NTSC games.
    The internal framebuffer on `env.console.bus.tia.framebuffer` is taller
    (312 rows, covers full PAL visible region after task #110); `get_screen`
    crops to the per-ROM `Display.YStart`/`Display.Height` window so the
    user-facing screen aligns vertically with xitari videos."""
    env = StellaEnvironment(_frame_loop_rom())
    env.reset()
    env.step(Action.NOOP)
    screen = env.get_screen()
    assert screen.shape == (210, 160)
    assert screen.dtype == jnp.uint8


def test_env_get_ram_returns_128_bytes():
    env = StellaEnvironment(_frame_loop_rom())
    env.reset()
    ram = env.get_ram()
    assert ram.shape == (128,)
    assert ram.dtype == jnp.uint8


def test_env_ale_aliases_exist():
    env = StellaEnvironment(_frame_loop_rom())
    env.reset()
    assert env.act is env.step or callable(env.act)
    env.act(Action.NOOP)
    # Matches `get_screen()`'s cropped (210, 160) shape — the ALE
    # `Display.YStart=34`/`Display.Height=210` default crop.
    assert env.getScreen().shape == (210, 160)
    assert env.getRAM().shape == (128,)
    assert env.gameOver() is False
    assert isinstance(env.getEpisodeFrameNumber(), int)


def test_env_action_propagates_to_ram_via_reader_rom():
    env = StellaEnvironment(_ram_reader_rom())
    env.reset()
    env.step(Action.LEFT)
    # RAM[$80] should reflect the LEFT button press.
    assert int(env.get_ram()[0]) == 0xBF


# --------------------------------------------------------------------------- #
# P6d — random NOOP reset (Mnih-style episode randomization)
# --------------------------------------------------------------------------- #

def test_env_reset_random_noop_zero_max_is_deterministic():
    """random_noop_max=0 leaves the frame counter at exactly the boot
    burn-in count (here, 0)."""
    env = StellaEnvironment(_frame_loop_rom())
    env.reset(random_noop_max=0)
    assert env.frame_number() == 0


def test_env_reset_random_noop_with_fixed_seed_is_reproducible():
    """A fixed seed produces the same frame_number across resets."""
    env1 = StellaEnvironment(_frame_loop_rom())
    env1.reset(random_noop_max=10, seed=42)
    n1 = env1.frame_number()

    env2 = StellaEnvironment(_frame_loop_rom())
    env2.reset(random_noop_max=10, seed=42)
    n2 = env2.frame_number()

    assert n1 == n2


def test_env_reset_random_noop_respects_max():
    """The post-reset frame_number is in [0, random_noop_max] inclusive."""
    for seed in range(8):
        env = StellaEnvironment(_frame_loop_rom())
        env.reset(random_noop_max=5, seed=seed)
        assert 0 <= env.frame_number() <= 5


def test_env_reset_random_noop_stacks_with_boot_burn():
    """random_noop_max is *additive* on top of boot_noop_steps +
    boot_reset_steps. With boot=3+0 and random_max=2/seed=fixed, the
    final frame_number is boot_noop_steps + boot_reset_steps + random."""
    env = StellaEnvironment(_frame_loop_rom())
    env.reset(boot_noop_steps=3, boot_reset_steps=0,
              random_noop_max=2, seed=0)
    n = env.frame_number()
    assert 3 <= n <= 5


def test_env_reset_rejects_negative():
    env = StellaEnvironment(_frame_loop_rom())
    import pytest
    with pytest.raises(ValueError):
        env.reset(random_noop_max=-1)


def test_env_reset_different_seeds_produce_different_states():
    """Sanity check: two different seeds with a non-trivial max should
    produce different starting frames (with overwhelming probability).
    This catches a regression where the seed is silently ignored."""
    starts = set()
    for seed in range(20):
        env = StellaEnvironment(_frame_loop_rom())
        env.reset(random_noop_max=30, seed=seed)
        starts.add(env.frame_number())
    # With 20 seeds drawing from {0..30} we expect well more than one
    # distinct outcome.
    assert len(starts) >= 3
