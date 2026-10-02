# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Phase 1: JAX backend reproduces ChemEx NumPy profiles under ``jax.jit``.

Covers the shipped examples whose experiment modules only needed the
output-stacking / matrix-power port.  Tolerance: max-abs error relative to the
max-abs signal ≤ 1e-9 (the NumPy eig/solve and JAX Padé propagators differ at
~1e-12).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.backend._examples import build_example
from tests.backend._jax_profile import local_values, make_local_function

pytestmark = pytest.mark.jax

PHASE1_EXAMPLES = [
    "CEST_13C",
    "CPMG_15N_IP",
    "CPMG_CHD2_1H_AP",
    "DCEST_15N",
    "DCEST_15N_HD_EXCH",
    "RELAXATION_HZNZ",
    "RELAXATION_NZ",
]
FORWARD_RTOL = 1e-9


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.max(np.abs(reference))) or 1.0
    return float(np.max(np.abs(np.asarray(actual) - reference))) / scale


@pytest.mark.parametrize("name", PHASE1_EXAMPLES)
def test_jit_forward_parity(name: str) -> None:
    import jax

    example = build_example(name)
    profiles = example.sample_profiles(3)
    assert len(profiles) >= min(3, len(example.profiles))
    for profile in profiles:
        reference = np.asarray(profile.calculate_unscaled(example.values), float)
        names, f = make_local_function(profile)
        x0 = np.array([local_values(profile, example.values)[n] for n in names])
        actual = np.asarray(jax.jit(f)(x0))
        assert actual.shape == reference.shape
        assert actual.dtype == np.float64
        error = _relative_error(actual, reference)
        assert error <= FORWARD_RTOL, (name, str(profile.spin_system), error)


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
