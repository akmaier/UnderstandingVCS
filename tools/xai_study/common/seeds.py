"""seeds — one place to make a P2 run reproducible.

Every experiment seeds Python's `random`, NumPy, and (optionally) the jaxtari
env's start-of-episode NOOP RNG from a single integer, and records *that* seed
in the result record (SPEC §R `seed`). Determinism in this project comes from
the emulator being exact under a fixed action trace; the seed only governs the
few genuinely-random choices (e.g. Mnih-style random-NOOP episode starts), so
keeping it in one helper keeps those choices auditable.

Usage:
    from tools.xai_study.common import seeds
    seeds.seed_everything(0)            # global Python + NumPy
    env.reset(..., seed=seeds.env_seed(0))   # env NOOP RNG, same family
"""
from __future__ import annotations

import os
import random
from typing import Optional

#: The default seed used across P2 when a caller does not pass one.
DEFAULT_SEED: int = 0


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed Python's `random`, NumPy, and `PYTHONHASHSEED`.

    Returns the seed so callers can record it directly in a result record.
    NumPy is seeded if importable (it always is in this venv); the import is
    guarded only so this module stays usable in a bare interpreter.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is always present here
        pass
    return seed


def env_seed(seed: Optional[int] = None) -> int:
    """Return the seed to pass to `StellaEnvironment.reset(seed=...)`.

    Centralised so the env's random-NOOP RNG is seeded from the same family as
    the global seed. `None` -> `DEFAULT_SEED` (deterministic).
    """
    return DEFAULT_SEED if seed is None else int(seed)
