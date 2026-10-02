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
