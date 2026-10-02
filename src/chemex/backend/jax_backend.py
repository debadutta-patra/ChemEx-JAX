# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Optional JAX backend (float64; compatible with ``jit``, ``vmap``, ``grad``).

Importing this module enables ``jax_enable_x64``: ChemEx's error bars are only
trustworthy in double precision (float32 gives falsely small uncertainties in
fast exchange), so float32 inputs are rejected rather than silently used.

Propagators
-----------
* Ordinary propagators use Padé-13 scaling and squaring (``expm`` below), which
  runs on CPU and GPU.  They differ from the NumPy backend's eigendecomposition
  path at the ~1e-12 level; that is expected.  ``jax.scipy.linalg.expm``
  (JAX 0.11.2) chooses ``floor(log2(|A|_1 / theta_13))`` squarings where
  Higham (2005) requires ``ceil``; the scaled norm can then reach
  ``2 theta_13`` and the result loses ~5 digits (2e-9 relative error for a
  D-CEST free-precession delay with |A|_1 = 19).  ``expm`` here scales with
  ``ceil`` itself and calls JAX's Padé step only on the scaled matrix.
* "Complete dephasing" propagators (oscillatory eigenmodes removed) need an
  eigendecomposition of a non-symmetric matrix.  JAX cannot differentiate
  such eigenvectors, so the propagator is a ``jax.custom_jvp`` using the
  Daleckii-Krein formula.  ``jnp.linalg.eig`` is CPU-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import expm as _jax_expm

jax.config.update("jax_enable_x64", True)

# ChemEx's oscillatory-mode threshold (chemex.nmr._pulses.propagators).
SMALL_VALUE = 1e-6
# Relative tolerance below which two eigenvalues are treated as degenerate in
# the Daleckii-Krein divided differences.
DEGENERACY_RTOL = 1e-9

# Padé-13 backward-error bound for float64 (Higham 2005, table 2.3).
THETA_13 = 5.371920351148152
# Upper bound on squarings: |A|_1 <= 2**20 * THETA_13 ≈ 5.6e6 (JAX defaults to
# 16, i.e. 3.5e5; the largest shipped example, COSCEST_1HN_IP_AP, needs 1.3e4).
MAX_SQUARINGS = 20


def expm(a: jax.Array) -> jax.Array:
    """Matrix exponential of one ``(n, n)`` float64 matrix (Higham 2005).

    Scaling uses ``ceil(log2(|A|_1 / theta_13))`` squarings; the Padé
    approximant itself is JAX's, applied to a matrix whose norm is within its
    accuracy bound (so JAX performs no squarings of its own).
    """
    norm = jnp.linalg.norm(a, 1)
    n_squarings = lax.stop_gradient(
        jnp.clip(jnp.ceil(jnp.log2(norm / THETA_13)), 0, MAX_SQUARINGS)
    )
    result = _jax_expm(a / 2.0**n_squarings, max_squarings=0)

    def square(carry: jax.Array, i: jax.Array) -> tuple[jax.Array, None]:
        return lax.cond(i < n_squarings, lambda x: x @ x, lambda x: x, carry), None

    result, _ = lax.scan(square, result, jnp.arange(MAX_SQUARINGS))
    # Beyond the squaring budget the result would be silently wrong: make it
    # visibly invalid instead.
    return jnp.where(norm <= 2.0**MAX_SQUARINGS * THETA_13, result, jnp.nan)


_expm_batched = jnp.vectorize(expm, signature="(n,n)->(n,n)")


def _require_float64(array: jax.Array, what: str) -> None:
    if array.dtype in (jnp.float32, jnp.complex64, jnp.float16, jnp.bfloat16):
        msg = (
            f"The JAX backend requires float64 inputs; {what} has dtype "
            f"{array.dtype}. Pass float64 values (float32 gives falsely small "
            "error bars in fast exchange)."
        )
        raise TypeError(msg)


def _dephased_parts(
    liouv: jax.Array, t: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    lam, vec = jnp.linalg.eig(liouv)
    inv = jnp.linalg.inv(vec)
    keep = jnp.abs(lam.imag) < SMALL_VALUE
    exp_lt = jnp.exp(lam * t)
    f = jnp.where(keep, exp_lt, 0.0)
    fprime = jnp.where(keep, t * exp_lt, 0.0)
    return lam, vec, inv, f, fprime


@jax.custom_jvp
def dephased_propagator(liouv: jax.Array, t: jax.Array) -> jax.Array:
    """``exp(L t)`` with oscillatory eigenmodes (|Im λ| >= 1e-6) removed.

    Mirrors the ``dephasing=True`` branch of ChemEx's ``calculate_propagators``
    for one positive duration ``t`` and one ``(n, n)`` Liouvillian.
    """
    _, vec, inv, f, _ = _dephased_parts(liouv, t)
    return ((vec * f) @ inv).real


@dephased_propagator.defjvp
def _dephased_propagator_jvp(
    primals: tuple[jax.Array, jax.Array], tangents: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    # Daleckii-Krein: dF = V (G o (V^-1 dL V)) V^-1, with G the matrix of
    # divided differences of f(λ) = mask * exp(λ t); f' on (near-)degenerate
    # pairs.  Durations are setup constants, so the t tangent is ignored.
    liouv, t = primals
    dliouv, _ = tangents
    lam, vec, inv, f, fprime = _dephased_parts(liouv, t)
    diff = lam[:, None] - lam[None, :]
    scale = 1.0 + jnp.abs(lam[:, None]) + jnp.abs(lam[None, :])
    close = jnp.abs(diff) <= DEGENERACY_RTOL * scale
    safe = jnp.where(close, 1.0, diff)
    g = jnp.where(
        close,
        0.5 * (fprime[:, None] + fprime[None, :]),
        (f[:, None] - f[None, :]) / safe,
    )
    dtilde = inv @ dliouv.astype(vec.dtype) @ vec
    out = ((vec * f) @ inv).real
    dout = (vec @ (g * dtilde) @ inv).real
    return out, dout


_dephased_batched = jnp.vectorize(dephased_propagator, signature="(n,n),()->(n,n)")


class JaxBackend:
    """JAX implementations of the evaluation-path kernels."""

    name = "jax"
    xp: ModuleType = jnp

    def propagators(
        self,
        liouv: Any,
        delays: float | Iterable[float],
        *,
        dephasing: bool = False,
    ) -> jax.Array:
        liouv = jnp.asarray(liouv)
        _require_float64(liouv, "the Liouvillian")
        # Delays are data/settings constants: keep them concrete NumPy so the
        # zero-duration and dephasing branches are resolved at trace time.
        delays_array = np.atleast_1d(np.asarray(delays, dtype=np.float64))
        if liouv.ndim == 2 and delays_array.size == 1 and not dephasing:
            return expm(liouv * delays_array[0])
        propagators = []
        for delay in delays_array:
            if dephasing and delay > 0:
                propagators.append(_dephased_batched(liouv, jnp.asarray(delay)))
            else:
                propagators.append(_expm_batched(liouv * delay))
        stacked = jnp.stack(propagators)
        return stacked[0] if stacked.shape[0] == 1 else stacked

    def matrix_power(self, a: Any, n: int) -> jax.Array:
        return jnp.linalg.matrix_power(jnp.asarray(a), int(n))

    def stack(self, seq: Sequence[Any]) -> jax.Array:
        return jnp.stack([jnp.asarray(item) for item in seq])

    def detect(
        self,
        detection_vector: Any,
        magnetization: Any,
        weights: Any,
    ) -> jax.Array:
        magnetization = jnp.asarray(magnetization)
        if (ndim := magnetization.ndim) >= 3:
            magnetization = (weights * magnetization).sum(axis=tuple(range(ndim - 2)))
        detected = jnp.asarray(detection_vector) @ magnetization
        if jnp.iscomplexobj(detected):
            detected = jnp.sign(detected.real) * jnp.abs(detected)
        return detected.reshape(())

    def evaluate_scientific_function(
        self,
        function_id: str,
        function: Callable[..., object],
        args: Sequence[object],
    ) -> object:
        return function(*args)

    def __repr__(self) -> str:
        return "JaxBackend()"

    def __reduce__(self) -> tuple[Callable[[str], object], tuple[str]]:
        from chemex.backend import get_backend

        return get_backend, (self.name,)

    def __copy__(self) -> JaxBackend:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> JaxBackend:
        return self


JAX_BACKEND = JaxBackend()
