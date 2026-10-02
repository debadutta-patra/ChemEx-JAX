# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Backend tests: ``pytest -m jax`` selects the JAX tests; they skip without JAX."""

from __future__ import annotations

from importlib.util import find_spec

import pytest

HAS_JAX = find_spec("jax") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "jax: needs the optional JAX backend (skipped when JAX is absent)"
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
