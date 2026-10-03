# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Array-backend protocol for ChemEx's evaluation path.

Setup code (configuration, data, basis and Liouvillian matrices) always builds
NumPy constants.  The evaluation path (``update`` -> ``l_free`` -> propagators
-> pulse-sequence ``calculate`` -> ``detect``) takes its backend from the
spectrometer's engine.  NumPy is the default and delegates verbatim to the
historical implementations, so its results are bit-for-bit unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

# Arrays on the evaluation path are NumPy or JAX arrays depending on the
# backend, so the protocol does not commit to either.
type BackendArray = Any


@runtime_checkable
class Backend(Protocol):
    """Numerical kernels used on the evaluation path."""

    @property
    def name(self) -> str:
        """Stable backend name (``"numpy"`` or ``"jax"``)."""
        ...

    @property
    def xp(self) -> ModuleType:
        """Array namespace (``numpy`` or ``jax.numpy``)."""
        ...

    def propagators(
        self,
        liouv: BackendArray,
        delays: float | Iterable[float],
        *,
        dephasing: bool = False,
    ) -> BackendArray:
        """Mirror ``chemex.nmr._pulses.propagators.calculate_propagators``."""
        ...

    def matrix_power(self, a: BackendArray, n: int) -> BackendArray:
        """Integer matrix power over the last two axes (``n`` is a constant)."""
        ...

    def stack(self, seq: Sequence[Any]) -> BackendArray:
        """Stack scalars or arrays along a new leading axis."""
        ...

    def detect(
        self,
        detection_vector: BackendArray,
        magnetization: BackendArray,
        weights: BackendArray,
    ) -> Any:
        """Detect the signal; a Python float on NumPy, a 0-d array on JAX."""
        ...

    def evaluate_scientific_function(
        self,
        function_id: str,
        function: Callable[..., object],
        args: Sequence[object],
    ) -> object:
        """Evaluate a bound scientific function (or its backend twin)."""
        ...
