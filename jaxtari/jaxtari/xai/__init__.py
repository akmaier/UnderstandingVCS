"""XAI attribution methods built on the differentiable VCS.

P8-a: Integrated Gradients on top of `jax.grad`.
P8-b: occlusion (ablation) attribution — the right primitive for the
      discrete-opcode SOFT-mode VCS, where naive zero-baseline IG
      misleads on the opcode bytes. See `occlusion.py` for the
      rationale.
"""

from jaxtari.xai.integrated_gradients import (
    assert_completeness,
    integrated_gradients,
)
from jaxtari.xai.occlusion import occlusion_attribution

__all__ = [
    "assert_completeness",
    "integrated_gradients",
    "occlusion_attribution",
]
