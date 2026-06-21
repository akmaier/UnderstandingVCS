"""replay — deterministic replay of an action trace to a target frame/state.

The emulator is exact under a fixed action sequence, so "a state" in P2 is
identified by *(game, boot convention, action trace, frame index)*. This module
turns that into a reproducible object: it steps a jaxtari env frame-by-frame and
returns the RAM / framebuffer at a requested frame (or the whole trajectory).

Conformance horizon (Paper-1): bit-exactness vs xitari holds only for the first
~30 frames of RAM and ~60 frames of screen after the standard boot (the
comparison videos are 60 fps, so frame = seconds x 60). `to_frame` enforces this
by default (`enforce_horizon=True`) so an experiment can't silently read a frame
where jaxtari and xitari may diverge; pass `enforce_horizon=False` to replay
further on purpose (e.g. behavioral probes that don't need bit-exactness).

Usage:
    from tools.xai_study.common import loader, replay
    env, _ = loader.load_game("pong")
    snap = replay.to_frame(env, actions=[0]*10, frame=10)   # ram + screen @ f10
    traj = replay.trajectory(env2, actions=[0]*20)          # all frames

Actions can be a list of ints or a path to an actions file (one int per line,
`#` comments allowed) in the same format the Paper-1 sweeps use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

#: Paper-1 conformance horizon (frames after the standard boot).
RAM_HORIZON_FRAMES: int = 30
SCREEN_HORIZON_FRAMES: int = 60

ActionsArg = Union[Sequence[int], str, os.PathLike]


@dataclass(frozen=True)
class Snapshot:
    """A single replayed frame's observable state."""
    frame: int                 # 1-based frame index reached (post that many steps)
    ram: np.ndarray            # (128,) uint8 RIOT RAM
    screen: np.ndarray         # (h, 160) uint8 palette indices
    actions: List[int]         # the prefix of actions applied to reach this frame


def load_actions(actions: ActionsArg) -> List[int]:
    """Coerce `actions` (list or file path) into a list of ints.

    File format matches the Paper-1 sweeps: one integer per line, blank lines and
    `#` comments ignored.
    """
    if isinstance(actions, (str, os.PathLike)):
        out: List[int] = []
        for line in Path(actions).read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(int(s))
        return out
    return [int(a) for a in actions]


def _ram(env) -> np.ndarray:
    return np.asarray(env.get_ram(), dtype=np.uint8)


def _screen(env) -> np.ndarray:
    return np.asarray(env.get_screen(), dtype=np.uint8)


def to_frame(env,
             actions: ActionsArg,
             frame: int,
             *,
             enforce_horizon: bool = True) -> Snapshot:
    """Step `env` `frame` times along `actions` and snapshot RAM + screen.

    `env` must already be reset (use `loader.load_game(..., reset=True)`, the
    default). `frame` is 1-based: `frame=10` applies the first 10 actions and
    returns the state after the 10th step.

    Raises ValueError if `frame` exceeds the action trace, or (when
    `enforce_horizon`) the screen horizon — so an experiment can't read past the
    bit-exact window unknowingly.
    """
    acts = load_actions(actions)
    if frame < 0:
        raise ValueError(f"frame must be >= 0, got {frame}")
    if frame > len(acts):
        raise ValueError(
            f"frame {frame} exceeds action trace length {len(acts)}")
    if enforce_horizon and frame > SCREEN_HORIZON_FRAMES:
        raise ValueError(
            f"frame {frame} is past the screen conformance horizon "
            f"({SCREEN_HORIZON_FRAMES}); pass enforce_horizon=False to replay "
            "further (bit-exactness vs xitari is not guaranteed there).")
    for a in acts[:frame]:
        env.step(int(a))
    return Snapshot(frame=frame, ram=_ram(env), screen=_screen(env),
                    actions=[int(a) for a in acts[:frame]])


def trajectory(env,
               actions: ActionsArg,
               *,
               n: Optional[int] = None,
               enforce_horizon: bool = True,
               collect_screen: bool = True) -> dict:
    """Replay the whole trace and return stacked per-frame arrays.

    Returns a dict with `frames` (n,), `ram` (n,128), and `screen` (n,h,160) if
    `collect_screen`. `n` caps the number of frames (default: full trace). With
    `enforce_horizon`, `n` is clamped to the screen horizon and a flag recorded.

    This is the small in-memory replay the unit test exercises; the heavyweight
    full-state recorder (RAM+registers+TIA+opcode+framebuffer) is E0-2's job.
    """
    acts = load_actions(actions)
    total = len(acts) if n is None else min(n, len(acts))
    clamped = False
    if enforce_horizon and total > SCREEN_HORIZON_FRAMES:
        total = SCREEN_HORIZON_FRAMES
        clamped = True
    rams = []
    screens = []
    for i in range(total):
        env.step(int(acts[i]))
        rams.append(_ram(env))
        if collect_screen:
            screens.append(_screen(env))
    out = {
        "frames": np.arange(1, total + 1, dtype=np.int64),
        "ram": np.stack(rams) if rams else np.empty((0, 128), np.uint8),
        "actions": np.asarray(acts[:total], dtype=np.int64),
        "horizon_clamped": clamped,
    }
    if collect_screen:
        out["screen"] = (np.stack(screens) if screens
                         else np.empty((0, 0, 160), np.uint8))
    return out
