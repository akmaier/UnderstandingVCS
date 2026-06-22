#!/usr/bin/env python3
"""P2-E6-2 benchmark — the TASK set.

The benchmark's tasks are the 6 core games (game_set.json §G) crossed with the
output regimes for which a committed §1 oracle map exists. Each task is one
(game, regime, target_output) with a fixed set of candidate causes; a method is
asked to attribute the target output to those causes, and is scored against the
oracle for that task (oracle.py + metrics.py).

The regimes carry the headline contrast (experiment_design.md §1):
  * ``content``  — a content/colour output (∂y/∂u defined; gradients can work);
  * ``position`` — a discrete sprite-position/index output whose NAIVE gradient
                   is zero, so only intervention/causal methods recover it;
  * ``ball_pixel`` — a content pixel on the ball band (a Pong running example).

A method that does well on ``content`` but collapses on ``position`` is exactly
the danger-zone story the leaderboard exposes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

from . import oracle as _oracle

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.dirname(HERE)
STUDY = os.path.dirname(COMPARE)
GAME_SET = os.path.join(STUDY, "common", "game_set.json")


@dataclass
class Task:
    game: str
    regime: str
    target_output: str
    n_causes: int

    @property
    def id(self) -> str:
        return f"{self.game}/{self.regime}"


def core_games() -> List[str]:
    """The 6 core games (game_set.json §G)."""
    try:
        gs = json.load(open(GAME_SET))
        return list(gs.get("core_set", []))
    except Exception:
        return ["pong", "breakout", "space_invaders",
                "seaquest", "ms_pacman", "qbert"]


def all_tasks() -> List[Task]:
    """Every benchmark task = (core game, regime) with a committed oracle map."""
    core = set(core_games())
    out: List[Task] = []
    for t in _oracle.available_tasks():
        if t["game"] not in core:
            continue
        out.append(Task(game=t["game"], regime=t["regime"],
                        target_output=t["target_output"],
                        n_causes=int(t["n_causes"])))
    return out


def get_task(game: str, regime: str) -> Task:
    for t in all_tasks():
        if t.game == game and t.regime == regime:
            return t
    raise KeyError(f"no benchmark task for game={game!r} regime={regime!r}; "
                   f"available: {[t.id for t in all_tasks()]}")


if __name__ == "__main__":
    for t in all_tasks():
        print(f"{t.id:<28} target={t.target_output:<26} n_causes={t.n_causes}")
