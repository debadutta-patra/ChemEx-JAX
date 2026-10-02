# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Complete-dephasing propagator: Daleckii-Krein ``custom_jvp`` at degenerate points.

The Liouvillians are CEST_15N's own (free evolution + B1 at one offset) with
the exchange parameters moved to degenerate limits.  JVPs along every local
parameter direction are compared with an exact reference: 34-digit central
differences of the mpmath dephased propagator (needs ``mpmath``).
"""

from __future__ import annotations

import numpy as np
import pytest

from chemex.backend import NUMPY_BACKEND
from tests.backend._examples import build_example
from tests.backend._jax_profile import local_values

pytestmark = pytest.mark.jax

JVP_RTOL = 1e-9
PRIMAL_RTOL = 1e-11


def _degenerate(values: dict[str, float], case: str) -> dict[str, float]:
    v = dict(values)
    if case in ("kex0", "all"):
        v["kab"] = v["kba"] = 0.0
    if case == "pb0":
        v["pa"], v["pb"], v["kab"] = 1.0, 0.0, 0.0
    if case in ("dw0", "all"):
        v["cs_i_b"] = v["cs_i_a"]
    if case in ("same_rates", "all"):
        v["r2_i_b"] = v["r2_i_a"]
        v["r1_i_b"] = v["r1_i_a"]
    return v


def _pulse_liouvillian(profile, values: dict[str, float], offset: float) -> np.ndarray:
    spectrometer = profile.spectrometer.with_backend(NUMPY_BACKEND)
    spectrometer.update(values)
    spectrometer.offset_i = offset
    engine = spectrometer._engine
    liouv = np.asarray(engine.l_free + engine.l_b1x_i, dtype=float)
    return liouv.reshape(liouv.shape[-2:])


@pytest.mark.parametrize("case", ["kex0", "pb0", "dw0", "same_rates", "all"])
def test_dephased_jvp_at_degenerate_points(case: str) -> None:
    pytest.importorskip("mpmath")
    import jax

    from chemex.backend.jax_backend import dephased_propagator
    from tests.backend._mp_backend import MpBackend

    example = build_example("CEST_15N")
    profile = example.profiles[0]
    assert profile.spectrometer.b1_i_distribution.dephasing
    t = float(profile.pulse_sequence.settings.time_t1)
    values = _degenerate(local_values(profile, example.values), case)
    liouv = _pulse_liouvillian(profile, values, offset=0.0)

    # Primal: the custom_jvp forward matches ChemEx's eig-path propagator.
    expected = NUMPY_BACKEND.propagators(liouv, [t, t], dephasing=True)[0]
    actual = np.asarray(dephased_propagator(liouv, t))
    assert np.max(np.abs(actual - expected)) <= PRIMAL_RTOL * np.max(np.abs(expected))

    mp_backend = MpBackend(dps=34)
    mp = mp_backend.mp
    for name in ("kab", "kba", "cs_i_a", "cs_i_b", "r2_i_a", "r2_i_b", "r1_i_b"):
        # The Liouvillian is linear in each parameter: an exact direction.
        bumped = dict(values)
        bumped[name] += 1.0
        direction = _pulse_liouvillian(profile, bumped, 0.0) - liouv
        if not np.any(direction):
            continue
        _, tangent = jax.jvp(dephased_propagator, (liouv, t), (direction, 0.0))
        tangent = np.asarray(tangent)
        assert np.all(np.isfinite(tangent)), (case, name)

        h = mp.mpf("1e-12")
        lm = np.vectorize(mp.mpf, otypes=[object])(liouv)
        dm = np.vectorize(mp.mpf, otypes=[object])(direction)
        plus = mp_backend._single(lm + h * dm, t, True)
        minus = mp_backend._single(lm - h * dm, t, True)
        exact = ((plus - minus) / (2 * h)).astype(float)
        scale = float(np.max(np.abs(exact))) or 1.0
        error = float(np.max(np.abs(tangent - exact))) / scale
        assert error <= JVP_RTOL, (case, name, error)
