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
