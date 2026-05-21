"""I/O abstraction — ALE-style actions, joystick + console-switch helpers.

See `jaxtari.io.action` for the public Action enum and the helper that
applies an action to a `Console`'s RIOT / TIA state.
"""

from jaxtari.io.action import (
    Action,
    NUM_ACTIONS,
    apply_action,
    console_switches,
)

__all__ = ["Action", "NUM_ACTIONS", "apply_action", "console_switches"]
