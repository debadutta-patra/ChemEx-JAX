# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Pure JAX functions of ChemEx profiles and residuals (optional ``jax`` extra).

Importing this package enables ``jax_enable_x64``; all inputs must be float64.

* :func:`compile_profile` — ``f(x) -> unscaled profile`` for one profile.
* :func:`compile_residuals` — ``r(x)`` = ChemEx's native weighted residuals
  (scaling, masks, error weighting, ordering), with ``chi2``.
* :func:`jacobian` — ``jacfwd``/``jacrev`` with optional memory-bounded chunks.

``x`` holds the values of ``free_ids`` (independent parameter ids of the
active parameterization); every other independent parameter is fixed at
``frame``.  The returned objects are pure and work with ``jax.jit``,
``jax.vmap``, ``jax.grad``/``jacfwd``/``jacrev``.  ChemEx's value-domain
checks cannot run on traced values: call ``.validate(x)`` on concrete inputs.

Long-lived processes: :func:`release_memory` drops compiled kernels and returns
freed memory to the OS (compilation is memory-hungry; see JAX_PORT_NOTES.md).
"""

from __future__ import annotations

from chemex.backend import get_backend

get_backend("jax")  # enables float64 before any JAX array is created

from chemex.jax._compile import (  # noqa: E402
    KERNEL_CACHE,
    CompiledProfile,
    CompiledResiduals,
    compile_profile,
    compile_residuals,
    jacobian,
    kernel_signature,
    release_memory,
)

__all__ = [
    "KERNEL_CACHE",
    "CompiledProfile",
    "CompiledResiduals",
    "compile_profile",
    "compile_residuals",
    "jacobian",
    "kernel_signature",
    "release_memory",
]
