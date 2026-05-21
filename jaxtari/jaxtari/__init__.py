"""jaxtari — differentiable JAX port of xitari.

See PORTING_PLAN.md at the repo root for scope and milestones.
"""

from jaxtari.types import CPUState
from jaxtari.diff.modes import Mode, current_mode, set_mode

__version__ = "0.0.1"

__all__ = ["CPUState", "Mode", "current_mode", "set_mode", "__version__"]
