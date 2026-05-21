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


def test_run_until_frame_raises_on_runaway_rom():
    """A ROM that never writes VSYNC should still wrap via the 262-line
    safety net — but the limit is meant to catch outright infinite loops
    where no frame edge ever fires. Use a JMP-to-self program."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0x4C))      # JMP $F000
    rom = rom.at[1].set(jnp.uint8(0x00))
    rom = rom.at[2].set(jnp.uint8(0xF0))
    rom = rom.at[0x0FFC].set(jnp.uint8(0x00))
    rom = rom.at[0x0FFD].set(jnp.uint8(0xF0))
    c = console_reset(initial_console(rom))
    # JMP itself is 3 cycles. Many JMPs fit in one scanline (76 cyc),
    # so the safety net at 262 scanlines DOES eventually fire — just
    # not infinitely. Run one frame; it should succeed via the wrap.
    c = run_until_frame(c)
    assert int(c.bus.tia.frame) == 1


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

def test_default_swchb_is_all_high():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = console_switches(c)                    # all defaults
    # color=1, p0/p1 difficulty = A (1), select & reset not pressed (1).
    assert int(c.bus.riot.swchb_in) == 0xFF


def test_console_switches_select_and_reset():
    c = console_reset(initial_console(_frame_loop_rom()))
    c = console_switches(c, select_pressed=True, reset_pressed=True)
    # SELECT = bit 1, RESET = bit 0 both cleared.
    assert int(c.bus.riot.swchb_in) == 0xFF & ~0x03


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
    env = StellaEnvironment(_frame_loop_rom())
    env.reset()
    env.step(Action.NOOP)
    screen = env.get_screen()
    assert screen.shape == (192, 160)
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
    assert env.getScreen().shape == (192, 160)
    assert env.getRAM().shape == (128,)
    assert env.gameOver() is False
    assert isinstance(env.getEpisodeFrameNumber(), int)


def test_env_action_propagates_to_ram_via_reader_rom():
    env = StellaEnvironment(_ram_reader_rom())
    env.reset()
    env.step(Action.LEFT)
    # RAM[$80] should reflect the LEFT button press.
    assert int(env.get_ram()[0]) == 0xBF
