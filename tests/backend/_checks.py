# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Numerical checks shared by the JAX parity tests and the report script.

Gradient criterion (from the port brief): compare ``jax.jacfwd`` with
Richardson-extrapolated central differences of the *NumPy* backend, keeping
the best of relative steps 1e-3, 1e-4, 1e-5.  A parameter is "strong" when its
effect ``max|dI/dθ| · s(θ) / max|I| >= 1e-6`` with ``s(θ) = |θ|`` (or 1 when
θ = 0, where a relative scale does not exist); strong parameters must agree to
1e-5 relative to ``max|dI/dθ|``.  Weak parameters are only cross-checked
against finite differences of the JAX function itself and reported.

Finite differences of a float64 function have a noise floor of roughly
``eval_noise / (effect * step)``; for weak-but-strong-classified parameters
(effect 1e-6 .. 1e-4) on eigendecomposition-based NumPy paths this exceeds
1e-5 even when ``jacfwd`` is exact.  When the NumPy-FD criterion fails for a
strong parameter, the check falls back to an exact reference: central
differences (relative step 1e-12) of the profile evaluated at 30 digits by
:class:`tests.backend._mp_backend.MpBackend`, requiring agreement to
``EXACT_RTOL = 1e-9`` on the ``EXACT_ROWS`` most sensitive points plus an even
spread (pulse sequences compute points independently; this is verified).  The
fallback needs ``mpmath``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np

from chemex.containers.data import Data
from chemex.containers.profile import Profile
from chemex.parameters.parameterization import ResolvedParameterValues
from tests.backend._jax_profile import local_values, make_local_function

FORWARD_RTOL = 1e-9
GRADIENT_RTOL = 1e-5
EFFECT_THRESHOLD = 1e-6
JACREV_RTOL = 1e-10
EXACT_RTOL = 1e-9
FD_STEPS = (1e-3, 1e-4, 1e-5)


def numpy_local_function(
    profile: Profile, names: tuple[str, ...]
) -> Callable[[np.ndarray], np.ndarray]:
    """NumPy backend as a function of the profile's local values."""
    metadata = np.array(profile.data.metadata, copy=True)

    def g(x: np.ndarray) -> np.ndarray:
        profile.spectrometer.update(dict(zip(names, map(float, x), strict=True)))
        data = Data(
            exp=np.zeros_like(profile.data.exp),
            err=np.zeros_like(profile.data.err),
            metadata=np.array(metadata, copy=True),
        )
        return np.asarray(profile.pulse_sequence.calculate(profile.spectrometer, data))

    return g


def relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.max(np.abs(reference))) or 1.0
    return float(np.max(np.abs(np.asarray(actual) - reference))) / scale


def richardson_column(
    function: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    k: int,
    rel_step: float,
) -> np.ndarray:
    scale = abs(x0[k]) if x0[k] != 0.0 else 1.0
    h = rel_step * scale

    def at(delta: float) -> np.ndarray:
        x = np.array(x0, dtype=float, copy=True)
        x[k] += delta
        return np.asarray(function(x), dtype=float)

    d_h = (at(h) - at(-h)) / (2.0 * h)
    d_h2 = (at(h / 2.0) - at(-h / 2.0)) / h
    return (4.0 * d_h2 - d_h) / 3.0


@dataclass(frozen=True)
class GradientRecord:
    name: str
    value: float
    effect: float
    error_vs_numpy: float
    error_vs_jax_fd: float | None
    error_vs_exact: float | None = None

    @property
    def strong(self) -> bool:
        return self.effect >= EFFECT_THRESHOLD

    @property
    def passes(self) -> bool:
        if not self.strong or self.error_vs_numpy <= GRADIENT_RTOL:
            return True
        return self.error_vs_exact is not None and self.error_vs_exact <= EXACT_RTOL


@dataclass(frozen=True)
class ProfileCheck:
    profile: str
    forward_error: float
    jacrev_error: float
    gradients: tuple[GradientRecord, ...]

    def worst_strong_gradient(self) -> float:
        return max((g.error_vs_numpy for g in self.gradients if g.strong), default=0.0)

    def failures(self) -> list[GradientRecord]:
        return [g for g in self.gradients if not g.passes]


EXACT_ROWS = 8
EXACT_DPS = 30


def _kernel_data(profile: Profile, rows: np.ndarray) -> Data:
    return Data(
        exp=np.zeros_like(profile.data.exp[rows]),
        err=np.zeros_like(profile.data.err[rows]),
        metadata=np.array(profile.data.metadata[rows], copy=True),
    )


def exact_rows(profile: Profile, columns: np.ndarray) -> np.ndarray:
    """Rows for the exact reference: largest |dI/dθ| per column plus a spread.

    Profile points are computed independently by the pulse sequences, so the
    reference can run on reduced metadata; :func:`exact_columns` verifies that
    in float64 and falls back to all rows otherwise.
    """
    size = profile.data.metadata.size
    picked = {int(np.argmax(np.abs(columns[:, k]))) for k in range(columns.shape[1])}
    picked |= {int(i) for i in np.linspace(0, size - 1, EXACT_ROWS).round()}
    return np.array(sorted(picked))


def exact_columns(
    profile: Profile,
    names: tuple[str, ...],
    x0: np.ndarray,
    indices: list[int],
    rows: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Exact Jacobian columns (on ``rows``) from 30-digit central differences.

    Returns the rows actually used and ``{index: column}``; needs mpmath.
    """
    from tests.backend._mp_backend import MpBackend

    full = numpy_local_function(profile, names)(x0)
    spectrometer = profile.spectrometer
    spectrometer.update(dict(zip(names, map(float, x0), strict=True)))
    subset = np.asarray(
        profile.pulse_sequence.calculate(spectrometer, _kernel_data(profile, rows))
    )
    if subset.shape != full[rows].shape or not np.allclose(
        subset, full[rows], rtol=1e-12, atol=1e-12 * np.max(np.abs(full))
    ):
        rows = np.arange(profile.data.metadata.size)

    backend = MpBackend(dps=EXACT_DPS)
    mp = backend.mp

    def evaluate(values: list[object]) -> np.ndarray:
        spectrometer = profile.spectrometer.with_backend(backend)  # type: ignore[arg-type]
        spectrometer.update(dict(zip(names, values, strict=True)))  # type: ignore[arg-type]
        return np.asarray(
            profile.pulse_sequence.calculate(spectrometer, _kernel_data(profile, rows)),
            dtype=object,
        )

    base = [mp.mpf(float(v)) for v in x0]
    columns = {}
    for k in indices:
        h = mp.mpf(abs(float(x0[k])) or 1.0) * mp.mpf("1e-12")
        plus, minus = list(base), list(base)
        plus[k] += h
        minus[k] -= h
        columns[k] = ((evaluate(plus) - evaluate(minus)) / (2 * h)).astype(float)
    return rows, columns


def check_profile(
    profile: Profile,
    values: ResolvedParameterValues,
    *,
    gradients: bool = True,
    exact: bool = True,
) -> ProfileCheck:
    import jax

    names, f = make_local_function(profile)
    x0 = np.array([local_values(profile, values)[n] for n in names])
    reference = np.asarray(profile.calculate_unscaled(values), float)
    actual = np.asarray(jax.jit(f)(x0))
    if actual.shape != reference.shape or actual.dtype != np.float64:
        msg = (
            f"shape/dtype mismatch: {actual.shape} {actual.dtype} vs {reference.shape}"
        )
        raise AssertionError(msg)
    forward = relative_error(actual, reference)
    if not gradients:
        return ProfileCheck(str(profile.spin_system), forward, 0.0, ())

    jac_fwd = np.asarray(jax.jit(jax.jacfwd(f))(x0))
    jac_rev = np.asarray(jax.jit(jax.jacrev(f))(x0))
    jac_scale = float(np.max(np.abs(jac_fwd))) or 1.0
    jacrev_error = float(np.max(np.abs(jac_fwd - jac_rev))) / jac_scale

    g = numpy_local_function(profile, names)
    f_np = jax.jit(f)
    signal = float(np.max(np.abs(reference))) or 1.0
    records = []
    for k, name in enumerate(names):
        column = jac_fwd[:, k]
        column_max = float(np.max(np.abs(column)))
        scale = abs(x0[k]) if x0[k] != 0.0 else 1.0
        effect = column_max * scale / signal
        if column_max == 0.0:
            records.append(GradientRecord(name, float(x0[k]), 0.0, 0.0, None))
            continue
        error = min(
            float(np.max(np.abs(column - richardson_column(g, x0, k, step))))
            / column_max
            for step in FD_STEPS
        )
        jax_fd = None
        if effect < EFFECT_THRESHOLD:
            jax_fd = min(
                float(
                    np.max(
                        np.abs(
                            column
                            - richardson_column(
                                lambda x: np.asarray(f_np(x)), x0, k, step
                            )
                        )
                    )
                )
                / column_max
                for step in FD_STEPS
            )
        records.append(GradientRecord(name, float(x0[k]), effect, error, jax_fd))
    # Exact-reference fallback for strong parameters whose NumPy FD is too noisy.
    pending = [
        k
        for k, r in enumerate(records)
        if r.strong and r.error_vs_numpy > GRADIENT_RTOL
    ]
    if pending and exact and find_spec("mpmath") is not None:
        rows = exact_rows(profile, jac_fwd[:, pending])
        rows, columns = exact_columns(profile, names, x0, pending, rows)
        for k, column in columns.items():
            scale = float(np.max(np.abs(column))) or 1.0
            error = float(np.max(np.abs(jac_fwd[rows, k] - column))) / scale
            records[k] = dataclasses.replace(records[k], error_vs_exact=error)
    # Restore the shared spectrometer to the reference values.
    profile.calculate_unscaled(values)
    return ProfileCheck(str(profile.spin_system), forward, jacrev_error, tuple(records))
