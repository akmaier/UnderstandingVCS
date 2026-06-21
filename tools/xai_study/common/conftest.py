"""pytest configuration for the xai_study harness tests.

Makes the package importable as both `tools.xai_study.common` (repo root on
path) and `xai_study.common` (the `tools/` dir on path), and exposes a `jaxtari`
availability marker so tests that need the emulator skip cleanly in an
environment without the jaxtari venv (CI without ROMs/JAX) instead of erroring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tools/xai_study/common/conftest.py -> repo root is parents[3].
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "tools"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _jaxtari_importable() -> bool:
    """True iff jax + the jaxtari package can be imported (the real emulator)."""
    # Put the (primary) jaxtari package dir on the path the same way loader does.
    try:
        from tools.xai_study.common import loader  # noqa: WPS433
    except Exception:
        return False
    loader._ensure_jaxtari_on_path()
    import importlib.util
    return (importlib.util.find_spec("jax") is not None
            and importlib.util.find_spec("jaxtari") is not None)


HAVE_JAXTARI = _jaxtari_importable()


@pytest.fixture(scope="session")
def have_jaxtari() -> bool:
    return HAVE_JAXTARI


requires_jaxtari = pytest.mark.skipif(
    not HAVE_JAXTARI,
    reason="jax/jaxtari not importable (run with the jaxtari venv python)")
