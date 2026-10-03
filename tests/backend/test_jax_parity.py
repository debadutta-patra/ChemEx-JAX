# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""JAX backend vs ChemEx NumPy: forward parity, gradients, transformations.

Cases: every shipped example run script (``examples/*/*/run*.sh``) plus a
synthetic configuration for each registered experiment type without a shipped
example (``tests/backend/synthetic``).  Functions take the profile's *local*
spectrometer parameter values; the constraint program is covered separately.

* Forward: 3 profiles spread over each case, ``jax.jit``, max-abs error /
  max-abs signal ≤ 1e-9.
* Gradients (first profile): ``jacfwd`` vs Richardson central differences of
  the NumPy backend (best of steps 1e-3, 1e-4, 1e-5) ≤ 1e-5 for every parameter
  with effect ≥ 1e-6, else vs an exact 34-digit reference ≤ 1e-9 (see
  ``_checks``); ``jacfwd`` vs ``jacrev`` ≤ 1e-10.
* Transformations: ``jit``, ``vmap`` over a parameter batch, ``grad`` of a χ².
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.backend._checks import (
    FORWARD_RTOL,
    JACREV_RTOL,
    check_profile,
)
from tests.backend._examples import (
    QUICK_CASES,
    all_cases,
    build_case,
    build_example,
    quick_or_full,
)
from tests.backend._jax_profile import local_values, make_local_function

pytestmark = pytest.mark.jax

CASES = quick_or_full(all_cases(), QUICK_CASES)


@pytest.mark.parametrize("case", CASES)
def test_forward_parity(case: str) -> None:
    example = build_case(case)
    profiles = example.sample_profiles(3)
    assert len(profiles) >= min(3, len(example.profiles))
    for profile in profiles:
        check = check_profile(profile, example.values, gradients=False)
        assert check.forward_error <= FORWARD_RTOL, (case, check)


@pytest.mark.parametrize("case", CASES)
def test_gradients(case: str) -> None:
    example = build_case(case)
    profile = example.profiles[0]
    check = check_profile(profile, example.values)
    assert check.forward_error <= FORWARD_RTOL, (case, check.forward_error)
    assert check.jacrev_error <= JACREV_RTOL, (case, check.jacrev_error)
    failures = check.failures()
    if failures and any(g.error_vs_exact is None for g in failures):
        pytest.importorskip("mpmath", reason="exact gradient reference needs mpmath")
    assert not failures, (case, failures)


@pytest.mark.parametrize("case", CASES)
def test_transformations(case: str) -> None:
    import jax
    import jax.numpy as jnp

    example = build_case(case)
    profile = example.profiles[0]
    names, f = make_local_function(profile)
    x0 = np.array([local_values(profile, example.values)[n] for n in names])

    batch = x0 * (1.0 + 1e-3 * np.arange(3)[:, None])
    batched = np.asarray(jax.jit(jax.vmap(f))(batch))
    single = np.stack([np.asarray(jax.jit(f)(x)) for x in batch])
    np.testing.assert_allclose(
        batched, single, rtol=0, atol=1e-12 * np.abs(single).max()
    )

    exp = jnp.asarray(profile.data.exp)
    err = np.asarray(profile.data.err)
    err = jnp.asarray(err if err.ndim == 1 else err.mean(axis=-1))
    mask = jnp.asarray(profile.data.mask)

    def chi2(x: jax.Array) -> jax.Array:
        calc = f(x)
        scale = jnp.sum(mask * calc * exp / err**2) / jnp.sum(mask * (calc / err) ** 2)
        scale = scale if profile.is_scaled else 1.0
        return jnp.sum(mask * ((scale * calc - exp) / err) ** 2)

    value, grad = jax.jit(jax.value_and_grad(chi2))(x0)
    assert np.isfinite(value)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_jax_then_numpy_leaves_numpy_unchanged() -> None:
    import jax

    example = build_example("CEST_13C")
    profile = example.profiles[0]
    before = np.array(profile.calculate_unscaled(example.values), copy=True)
    names, f = make_local_function(profile)
    x0 = np.array([local_values(profile, example.values)[n] for n in names])
    jax.jit(f)(x0).block_until_ready()
    jax.jacfwd(f)(x0).block_until_ready()
    jax.jit(jax.vmap(f))(np.stack([x0, x0])).block_until_ready()
    assert profile.spectrometer.backend.name == "numpy"
    assert all(
        type(v) is not jax.Array for v in profile.spectrometer.par_values.values()
    )
    after = np.asarray(profile.calculate_unscaled(example.values))
    np.testing.assert_array_equal(after, before)


def test_float32_inputs_are_rejected() -> None:
    import jax.numpy as jnp

    from chemex.backend import get_backend

    backend = get_backend("jax")
    with pytest.raises(TypeError, match="float64"):
        backend.propagators(jnp.eye(4, dtype=jnp.float32), [0.1, 0.2])
