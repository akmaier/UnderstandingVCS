"""TIA — Television Interface Adapter (Atari 2600 video / audio chip).

See `jaxtari.tia.system` for the type, address constants, and peek/poke.
"""

from jaxtari.tia.system import (
    COLOR_CLOCKS_PER_CPU_CYCLE,
    COLOR_CLOCKS_PER_SCANLINE,
    HBLANK_COLOR_CLOCKS,
    NTSC_CPU_CYCLES_PER_SCANLINE,
    NTSC_SCANLINES_PER_FRAME,
    NUM_REGISTERS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TIAState,
    VISIBLE_HEIGHT,
    Y_START,
    initial_tia_state,
    render_pixel,
    render_playfield_scanline,
    render_scanline,
    set_paddle,
    set_paddle_resistance,
    set_trigger,
    tia_advance,
    tia_apply_wsync,
    tia_peek,
    tia_poke,
)

__all__ = [
    "COLOR_CLOCKS_PER_CPU_CYCLE",
    "COLOR_CLOCKS_PER_SCANLINE",
    "HBLANK_COLOR_CLOCKS",
    "NTSC_CPU_CYCLES_PER_SCANLINE",
    "NTSC_SCANLINES_PER_FRAME",
    "NUM_REGISTERS",
    "SCREEN_HEIGHT",
    "SCREEN_WIDTH",
    "TIAState",
    "VISIBLE_HEIGHT",
    "Y_START",
    "initial_tia_state",
    "render_pixel",
    "render_playfield_scanline",
    "render_scanline",
    "set_paddle",
    "set_paddle_resistance",
    "set_trigger",
    "tia_advance",
    "tia_apply_wsync",
    "tia_peek",
    "tia_poke",
]
