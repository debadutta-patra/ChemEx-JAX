# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Phase-1/2 test helper: a profile as a JAX function of its local values.

The function differentiates with respect to the profile's *local* spectrometer
parameter values (``profile.name_map`` targets).  The public
``chemex.jax.compile_profile`` (Phase 4) adds the constraint program on top.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from chemex.backend import get_backend
from chemex.containers.data import Data
from chemex.containers.profile import Profile


def local_values(profile: Profile, values: Mapping[str, float]) -> dict[str, float]:
    return {local: float(values[pid]) for local, pid in profile.name_map.items()}


def make_local_function(
    profile: Profile,
) -> tuple[tuple[str, ...], Callable[..., object]]:
    """Return ``(names, f)`` with ``f(x) -> unscaled profile`` on the JAX backend.

    Each call (each trace) works on a private deep copy of the profile's
    spectrometer, so traced values never reach the shared NumPy spectrometer.
    """
    import jax.numpy as jnp

    backend = get_backend("jax")
    names = tuple(profile.name_map)
    metadata = np.array(profile.data.metadata, copy=True)

    def f(x: object) -> object:
        spectrometer = profile.spectrometer.with_backend(backend)
        spectrometer.update(dict(zip(names, list(x), strict=True)))  # type: ignore[arg-type]
        kernel_data = Data(
            exp=np.zeros_like(profile.data.exp),
            err=np.zeros_like(profile.data.err),
            metadata=metadata,
        )
        return jnp.asarray(profile.pulse_sequence.calculate(spectrometer, kernel_data))

    return names, f
