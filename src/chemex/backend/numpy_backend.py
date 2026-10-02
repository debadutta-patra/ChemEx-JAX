# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Default NumPy backend: verbatim delegation to the historical kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any

import numpy as np
from numpy.linalg import matrix_power

from chemex.typing import Array


class NumpyBackend:
    """Backend whose every operation is the upstream NumPy implementation.

    ``chemex.nmr`` is imported lazily inside the methods because the NMR
    engine imports this module; the backend package must stay a leaf.
    """

    name = "numpy"
    xp: ModuleType = np

    def propagators(
        self,
        liouv: Array,
        delays: float | Iterable[float],
        *,
        dephasing: bool = False,
    ) -> Array:
        from chemex.nmr._pulses.propagators import calculate_propagators

        return calculate_propagators(liouv, delays, dephasing=dephasing)

    def matrix_power(self, a: Array, n: int) -> Array:
        return matrix_power(a, n)

    def stack(self, seq: Sequence[Any]) -> Array:
        return np.array(seq)

    def detect(
        self,
        detection_vector: Array,
        magnetization: Array,
        weights: Array,
    ) -> float:
        from chemex.nmr._engine.magnetization import detect_signal

        return detect_signal(detection_vector, magnetization, weights)

    def evaluate_scientific_function(
        self,
        function_id: str,
        function: Callable[..., object],
        args: Sequence[object],
    ) -> object:
        return function(*args)

    def __repr__(self) -> str:
        return "NumpyBackend()"

    # Backends are stateless singletons: copy and pickle them by name so that
    # deep-copied spectrometers and worker processes share the same instance.
    def __reduce__(self) -> tuple[Callable[[str], object], tuple[str]]:
        from chemex.backend import get_backend

        return get_backend, (self.name,)

    def __copy__(self) -> NumpyBackend:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> NumpyBackend:
        return self


NUMPY_BACKEND = NumpyBackend()
