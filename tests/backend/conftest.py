# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Backend tests: ``pytest -m jax`` selects the JAX tests; they skip without JAX."""

from __future__ import annotations

import fcntl
import tempfile
from importlib.util import find_spec
from pathlib import Path

import pytest

HAS_JAX = find_spec("jax") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "jax: needs the optional JAX backend (skipped when JAX is absent)"
    )
    config.addinivalue_line(
        "markers",
        "memory_heavy: needs several GB; such tests run one at a time across "
        "xdist workers (inter-process lock)",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if HAS_JAX:
        return
    skip = pytest.mark.skip(reason="JAX is not installed (pip install 'chemex[jax]')")
    for item in items:
        if item.get_closest_marker("jax") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _enable_jax_x64(request: pytest.FixtureRequest) -> None:
    """Import the JAX backend (which enables float64) before any JAX array exists.

    ``jax_enable_x64`` only affects arrays created after it is set; importing
    the backend lazily inside a traced function would trace in float32.
    """
    if HAS_JAX and request.node.get_closest_marker("jax") is not None:
        from chemex.backend import get_backend

        get_backend("jax")


@pytest.fixture(autouse=True)
def _release_jax_caches(request: pytest.FixtureRequest):
    """Bound per-worker memory: drop compiled kernels after each JAX test."""
    yield
    if not HAS_JAX or request.node.get_closest_marker("jax") is None:
        return
    import sys

    if "chemex.jax" in sys.modules:
        sys.modules["chemex.jax"].release_memory()
    else:
        import jax

        jax.clear_caches()


_HEAVY_LOCK = Path(tempfile.gettempdir()) / "chemex-jax-memory-heavy.lock"


@pytest.fixture(autouse=True)
def _serialize_memory_heavy(request: pytest.FixtureRequest):
    """Let only one ``memory_heavy`` test run at a time, whatever ``-n`` is.

    Compiling a 468-profile D-CEST example or a whole-example ``jacrev`` peaks
    at 5-8 GB; several at once on different xdist workers exhausted memory.
    """
    if request.node.get_closest_marker("memory_heavy") is None:
        yield
        return
    with _HEAVY_LOCK.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
