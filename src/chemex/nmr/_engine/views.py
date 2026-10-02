# Modified in the ChemEx-JAX fork (GPL-3.0-or-later): method-form squeeze
# so JAX arrays pass through.

from __future__ import annotations

from chemex.typing import Array


def reshape_single_liouvillian(
    liouvillian: Array,
    size: int,
    *,
    purpose: str,
) -> Array:
    squeezed = liouvillian.squeeze()
    expected_shape = (size, size)
    if squeezed.shape != expected_shape:
        msg = (
            f"{purpose} requires single-point B1 and Jeff distributions; "
            f"got Liouvillian shape {liouvillian.shape} "
            f"(squeezed to {squeezed.shape})."
        )
        raise ValueError(msg)
    return squeezed.reshape(expected_shape)
