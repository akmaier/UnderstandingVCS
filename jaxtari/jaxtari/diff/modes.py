"""Global HARD vs SOFT execution mode toggle.

Paper reference: the execution-mode toggle behind the "three execution
modes" of the paper (supplementary Table "The three execution modes").
HARD is the bit-exact conformance machine; SOFT here is the SOFT-STE
mode used for every gradient in the paper — its forward pass is
bit-identical to HARD (Theorem 1, "Exact forward equivalence") and its
backward pass is a surrogate (Corollary 1). The third, fully relaxed
mode (FULL, used only for the T -> 0 study of Theorem 2) is not a
separate enum value; it is reached by enabling the default-off forward
relaxation hook (see RelaxConfig in the jutari twin / the relaxation
study).

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
