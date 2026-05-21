"""ALE-style environment wrapper for the differentiable simulator.

See `jaxtari.env.stella_environment` for the `StellaEnvironment`
class — `reset` / `step(action)` / `get_screen` / `get_ram` / etc.
"""

from jaxtari.env.stella_environment import StellaEnvironment

__all__ = ["StellaEnvironment"]
