# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""High-precision reference backend (mpmath on NumPy object arrays).

Test/diagnostic oracle only: runs ChemEx's unmodified sequence code with
propagators evaluated by ``mpmath`` at ``dps`` digits, so central differences
with tiny steps give essentially exact derivatives.  Requires ``mpmath``
(a SymPy dependency; not a ChemEx dependency).  Slow: use on single profiles.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any

import numpy as np

SMALL_VALUE = 1e-6


class MpBackend:
    name = "mpmath"
    xp: ModuleType = np

    def __init__(self, dps: int = 30) -> None:
        import mpmath

        self.mp = mpmath.mp.clone()
        self.mp.dps = dps

    def _to_mp(self, matrix: np.ndarray) -> Any:
        return self.mp.matrix([[self.mp.mpf(v) for v in row] for row in matrix])

    def _from_mp(self, matrix: Any, n: int) -> np.ndarray:
        out = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                out[i, j] = value.real if isinstance(value, self.mp.mpc) else value
        return out

    def _single(self, liouv: np.ndarray, delay: float, dephasing: bool) -> np.ndarray:
        n = liouv.shape[-1]
        lm = self._to_mp(liouv)
        t = self.mp.mpf(delay)
        if dephasing and delay > 0:
            eigenvalues, vectors = self.mp.eig(lm)
            weights = [
                self.mp.exp(lam * t) if abs(self.mp.im(lam)) < SMALL_VALUE else 0
                for lam in eigenvalues
            ]
            result = vectors * self.mp.diag(weights) * self.mp.inverse(vectors)
        else:
            result = self.mp.expm(lm * t)
        return self._from_mp(result, n)

    def propagators(
        self,
        liouv: Any,
        delays: float | Iterable[float],
        *,
        dephasing: bool = False,
    ) -> np.ndarray:
        liouv = np.asarray(liouv, dtype=object)
        delays_array = np.atleast_1d(np.asarray(delays, dtype=np.float64))
        lead = liouv.shape[:-2]
        n = liouv.shape[-1]
        out = np.empty((delays_array.size, *lead, n, n), dtype=object)
        for d, delay in enumerate(delays_array):
            for index in np.ndindex(*lead):
                out[(d, *index)] = self._single(liouv[index], float(delay), dephasing)
        return out[0] if out.shape[0] == 1 else out

    def matrix_power(self, a: Any, n: int) -> np.ndarray:
        a = np.asarray(a, dtype=object)
        result = np.broadcast_to(np.eye(a.shape[-1], dtype=object), a.shape).copy()
        base = a
        while n:
            if n & 1:
                result = result @ base
            base = base @ base
            n >>= 1
        return result

    def stack(self, seq: Sequence[Any]) -> np.ndarray:
        items = [np.asarray(item, dtype=object) for item in seq]
        return np.stack(items) if items else np.empty(0, dtype=object)

    def detect(self, detection_vector: Any, magnetization: Any, weights: Any) -> Any:
        magnetization = np.asarray(magnetization, dtype=object)
        if (ndim := magnetization.ndim) >= 3:
            magnetization = (weights * magnetization).sum(axis=tuple(range(ndim - 2)))
        return (np.asarray(detection_vector) @ magnetization).reshape(())[()]

    def evaluate_scientific_function(
        self,
        function_id: str,
        function: Callable[..., object],
        args: Sequence[object],
    ) -> object:
        return function(*args)
