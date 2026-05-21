"""Global HARD vs SOFT execution mode toggle.

HARD: bit-exact emulation against xitari. Integer state, hard opcode switch,
indexed memory reads. No gradients. This is the default and what every
conformance test in `tests/conformance/` runs in.

SOFT: relaxed differentiable emulation. Float state, softmax opcode dispatch,
NTM-style soft memory reads/writes, ROM-as-weights. Gradients flow.

The mode is a process-global so the same modules can be reused in both paths
without threading a context through every function. Use `set_mode` /
`current_mode` to switch, or the `using_mode` context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Iterator


class Mode(Enum):
    HARD = "hard"
    SOFT = "soft"


_current: Mode = Mode.HARD


def current_mode() -> Mode:
    return _current


def set_mode(mode: Mode) -> None:
    global _current
    _current = mode


@contextmanager
def using_mode(mode: Mode) -> Iterator[None]:
    """`with using_mode(Mode.SOFT): ...` — scoped mode switch."""
    global _current
    previous = _current
    _current = mode
    try:
        yield
    finally:
        _current = previous
