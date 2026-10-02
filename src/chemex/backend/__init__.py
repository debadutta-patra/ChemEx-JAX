# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Array backends for ChemEx's evaluation path (NumPy default, optional JAX).

``import chemex.backend`` never imports JAX.  ``get_backend("jax")`` imports
:mod:`chemex.backend.jax_backend` on first use, which requires the optional
``jax`` extra and enables float64 (``jax_enable_x64``).
"""

from __future__ import annotations

from importlib.util import find_spec

from chemex.backend.base import Backend
from chemex.backend.numpy_backend import NUMPY_BACKEND, NumpyBackend

__all__ = ["NUMPY_BACKEND", "Backend", "NumpyBackend", "get_backend"]


def get_backend(name: str = "numpy") -> Backend:
    """Return the backend singleton called ``name`` (``"numpy"`` or ``"jax"``)."""
    if name == "numpy":
        return NUMPY_BACKEND
    if name == "jax":
        if find_spec("jax") is None:
            msg = (
                "The JAX backend needs the optional dependency: "
                "pip install 'chemex[jax]'"
            )
            raise ImportError(msg)
        from chemex.backend.jax_backend import JAX_BACKEND

        return JAX_BACKEND
    msg = f"Unknown backend {name!r}; expected 'numpy' or 'jax'"
    raise ValueError(msg)
