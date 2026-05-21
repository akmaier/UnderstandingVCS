"""RIOT — MOS 6532 (RAM/I/O/Timer) chip.

The 128 B RAM is already wired into the Bus directly (P2). This module
covers the timer + I/O ports that live at \$0280–\$029F.
"""

from jaxtari.riot.system import (
    RIOTState,
    initial_riot_state,
    riot_advance,
    riot_peek,
    riot_poke,
    set_swcha_input,
    set_swchb_input,
)

__all__ = [
    "RIOTState",
    "initial_riot_state",
    "riot_advance",
    "riot_peek",
    "riot_poke",
    "set_swcha_input",
    "set_swchb_input",
]
