#!/usr/bin/env python3
"""P2-E6-2 benchmark — EXAMPLE methods (the plug-in contract).

A benchmark method is any callable

    method(task, oracle) -> {"attribution": {cause_name: float},
                              "S": float|None, "M": float|None}

`attribution` scores every candidate cause (higher = more important); `S` and
`M` are the optional sufficiency / minimality legs the method reports (the
benchmark measures F itself). Plug your own method in by passing
``--method dotted.path:callable`` to ``run.py``; these examples define the two
ends of the scale a real method should fall between.

  * ``oracle_copy``  — a positive control: copies the oracle's own importance.
                       Must score faithfulness == 1 / precision@k == 1 (this is
                       how the benchmark validates its own scoring pipeline,
                       mirroring the committed "oracle-as-method corr=1" control).
  * ``uniform``      — a trivial baseline: equal score to every cause. Constant
                       attribution ⇒ corr 0 (the floor a real method must beat).
  * ``random``       — seeded random scores: a chance baseline.
  * ``magnitude_proxy`` — a deterministic, oracle-free toy "gradient-like" method
                       that scores causes by a cheap structural prior (RAM index
                       recency) — it has NO access to the oracle, so it lands
                       somewhere between the floor and the ceiling, demonstrating
                       a genuine end-to-end score for a method that is not the
                       oracle.
"""
from __future__ import annotations

import random
from typing import Dict


def oracle_copy(task, oracle) -> Dict:
    """Positive control: the oracle scoring itself ⇒ perfect faithfulness."""
    return {"attribution": dict(zip(oracle.cause_names, oracle.abs_delta)),
            "S": 1.0, "M": 1.0}


def uniform(task, oracle) -> Dict:
    """Constant attribution ⇒ corr 0 (the floor)."""
    return {"attribution": {n: 1.0 for n in oracle.cause_names}}


def random_method(task, oracle, seed: int = 0) -> Dict:
    rng = random.Random(seed + hash(task.id) % 10_000)
    return {"attribution": {n: rng.random() for n in oracle.cause_names}}


def magnitude_proxy(task, oracle) -> Dict:
    """An oracle-free toy method. Scores each cause by a cheap, deterministic
    structural prior: parse the RAM index out of names like 'ram[45]:set' and use
    it as a (fake) saliency, plus a small bonus for ':set' over ':occlude'. It
    knows NOTHING about the true Δy, so its score is a genuine, non-trivial
    end-to-end result — exactly what plugging in a real attribution method
    produces."""
    attr: Dict[str, float] = {}
    for n in oracle.cause_names:
        score = 0.0
        if n.startswith("ram[") and "]" in n:
            try:
                idx = int(n[4:n.index("]")])
                score = (idx % 64) / 64.0  # bounded structural prior
            except ValueError:
                score = 0.1
        elif n.startswith("joystick"):
            score = 0.5
        elif n.startswith("tia"):
            score = 0.3
        if n.endswith(":set"):
            score += 0.05
        attr[n] = score
    return {"attribution": attr}


# the registry the CLI exposes by short name
EXAMPLES = {
    "oracle_copy": oracle_copy,
    "uniform": uniform,
    "random": random_method,
    "magnitude_proxy": magnitude_proxy,
}
