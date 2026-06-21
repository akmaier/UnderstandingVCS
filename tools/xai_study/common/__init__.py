"""xai_study.common — shared harness for Paper 2 (ground-truth interpretability).

The small, stable foundation every later P2 experiment imports (SPEC §E0-1):

- `loader`   — resolve a game *name* to a ROM file and build a jaxtari
               `StellaEnvironment` for it (xitari-parity boot).
- `replay`   — deterministic replay of an action trace to a target frame /
               state, within the Paper-1 conformance horizon.
- `results`  — the self-describing results-record writer/reader (SPEC §R):
               one `<exp>_<game>[_<state>].json` per record, with array payloads
               in a sibling `.npz`.
- `seeds`    — one place to seed Python / NumPy / the env's NOOP RNG so a run
               is reproducible and the seed is recorded in every result.

Import as a package:

    from tools.xai_study.common import loader, replay, results, seeds

or, with `tools/` (the repo root's `tools` parent) on `PYTHONPATH`:

    from xai_study.common import loader, replay, results, seeds

Both forms work because this package has no implicit-relative imports.
"""
from __future__ import annotations

from . import loader, replay, results, seeds  # noqa: F401

__all__ = ["loader", "replay", "results", "seeds"]
