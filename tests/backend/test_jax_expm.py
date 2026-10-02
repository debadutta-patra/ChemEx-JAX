# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""The JAX backend's ``expm`` scales with Higham's ``ceil`` rule.

``jax.scipy.linalg.expm`` (0.11.2) uses ``floor`` and loses ~5 digits when
``|A|_1 / theta_13`` is just below a power of two; ChemEx parity at 1e-9 needs
the corrected scaling.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

pytestmark = pytest.mark.jax


def _bloch_mcconnell_like(rng: np.random.Generator, n: int, norm: float) -> np.ndarray:
    # Weakly damped rotations plus exchange: the regime of ChemEx delays.
    a = rng.normal(size=(n, n))
    a = (a - a.T) + 0.01 * rng.normal(size=(n, n)) - 0.05 * np.eye(n)
    return a * (norm / np.linalg.norm(a, 1))


@pytest.mark.parametrize("norm", [0.0, 1e-3, 0.5, 5.0, 5.4, 10.7, 19.4, 40.0, 1e3])
def test_expm_matches_scipy(norm: float) -> None:
    from chemex.backend.jax_backend import expm

    rng = np.random.default_rng(1234)
    a = _bloch_mcconnell_like(rng, 6, norm) if norm else np.zeros((6, 6))
    expected = scipy.linalg.expm(a)
    actual = np.asarray(expm(a))
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(actual - expected)) / scale <= 1e-12


def test_expm_beyond_squaring_budget_is_nan() -> None:
    from chemex.backend.jax_backend import MAX_SQUARINGS, THETA_13, expm

    a = np.eye(4) * (2.0**MAX_SQUARINGS * THETA_13 * 2)
    assert np.all(np.isnan(np.asarray(expm(a))))


def test_expm_gradient_is_finite_at_zero() -> None:
    import jax
    import jax.numpy as jnp

    from chemex.backend.jax_backend import expm

    grad = jax.grad(lambda x: jnp.sum(expm(x * jnp.eye(3))))(0.0)
    np.testing.assert_allclose(grad, 3.0, rtol=1e-14)
