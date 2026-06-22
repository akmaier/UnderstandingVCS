"""P2-E6-2 — the reusable P2 interpretability benchmark (tasks + oracle + metrics).

Package a third party can use to score one interpretability method end-to-end
against the P2 ground-truth oracle and get a comparable faithfulness score.

  * ``tasks``          — the TASK set (6 core games × output regimes).
  * ``oracle``         — the ORACLE interface (the §1 ground-truth causal map).
  * ``metrics``        — the METRIC definitions (corr / del-ins AUC / p@k / F-S-M).
  * ``run``            — the runnable entry point (``python -m ...run``).
  * ``example_method`` — bundled plug-in examples (positive control + baselines).

See README.md for the full contract and the runnable example. The oracle ground
truth is read from the committed §R records; no ROM is required to score a
method (ROM handling for the optional live AUC is in rom_manifest.json).
"""

__all__ = ["tasks", "oracle", "metrics", "run", "example_method"]
__version__ = "1.0.0"
