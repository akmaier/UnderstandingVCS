"""pytest conftest — pin JAX / XLA / BLAS to a single thread BEFORE any test
module imports jax.

jaxtari runs eager on the CPU with tiny (≤160-element) arrays, so XLA/BLAS
multi-threading only adds overhead. `tools/jaxtari_dump.py` (the sweep worker)
already pins single-threaded for exactly this reason. The test suite did NOT —
so under pytest-xdist (`-n auto`, see pyproject `addopts`) each spawned worker
span up its own multi-threaded JAX/XLA, oversubscribing the machine; a worker
could then be killed mid-test and the controller would deadlock (0% CPU, no
workers) → in CI the jaxtari job is "cancelled". Pinning single-threaded here
removes that oversubscription and makes per-op work faster.

This must run before jax is imported. pytest imports conftest.py during
collection, BEFORE it imports the test modules (which import jax), and each
xdist worker re-imports it — so setting the env at module top covers both the
controller and every worker. (conftest itself must not import jax.)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_xla = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _xla:
    os.environ["XLA_FLAGS"] = (_xla + " --xla_cpu_multi_thread_eigen=false").strip()


# --- Heavy-suite memory bound (opt-in via XAI_CLEAR_JAX_CACHES=1) ------------
# The JAX-autodiff suites (P7 functional / P8 IG) accumulate compiled-function
# and traced-array memory across tests; over a long heavy run a SINGLE worker's
# RSS climbs past the 16 GB CI runner and the job is OOM-killed ("the runner has
# received a shutdown signal", exit 143) — even at `-n 1`. When the flag is set
# (heavy.yml sets it on the autodiff matrix groups) we release the JAX caches +
# force GC after every test, bounding the peak. OFF by default so the fast PR
# gate (test.yml) pays no per-test recompile cost. conftest must not import jax
# at module top (see above), so the import is lazy, inside the fixture.
if os.environ.get("XAI_CLEAR_JAX_CACHES"):
    import gc
    import pytest

    @pytest.fixture(autouse=True)
    def _clear_jax_caches_after_test():
        yield
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()
