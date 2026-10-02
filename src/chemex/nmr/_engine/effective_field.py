# Modified in the ChemEx-JAX fork (GPL-3.0-or-later): effective-field tilts
# accept an array namespace (traced angles, functional updates).

"""Helpers for effective-field rotations in NMR Liouvillian calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from chemex.nmr.basis import Basis
from chemex.typing import Array


@dataclass(frozen=True, slots=True)
class EffectiveFieldTilt:
    """Indices and angle for tilting one state's Ix/Iz pair."""

    index_x: int
    index_z: int
    angle: float


def calculate_i_effective_field_angle(
    *,
    b1_i: float,
    cs_i: float,
    ppm_i: float,
    carrier_i: float,
    offset_i: float,
    xp: ModuleType = np,
) -> float:
    """Calculate the tilt angle between the z-axis and the effective field."""
    w1 = b1_i * 2.0 * np.pi
    wi = -(cs_i * ppm_i - carrier_i * ppm_i - offset_i * 2.0 * np.pi * np.sign(ppm_i))
    if xp is not np:
        return xp.arctan2(w1, wi)
    return float(np.arctan2(w1, wi))


def build_i_effective_field_tilts(
    basis: Basis,
    states: Iterable[str],
    par_values: dict[str, float],
    *,
    b1_i: float,
    ppm_i: float,
    carrier_i: float,
    offset_i: float,
    xp: ModuleType = np,
) -> tuple[EffectiveFieldTilt, ...]:
    """Build per-state Ix/Iz tilts for rotation along the effective field."""
    component_indices = {
        component_state: index
        for index, component_state in enumerate(basis.components_states)
    }
    return tuple(
        EffectiveFieldTilt(
            index_x=component_indices[f"ix_{state}"],
            index_z=component_indices[f"iz_{state}"],
            angle=calculate_i_effective_field_angle(
                b1_i=b1_i,
                cs_i=par_values[f"cs_i_{state}"],
                ppm_i=ppm_i,
                carrier_i=carrier_i,
                offset_i=offset_i,
                xp=xp,
            ),
        )
        for state in states
    )


def tilt_magnetization_along_i_effective_field(
    magnetization: Array,
    tilts: Iterable[EffectiveFieldTilt],
    *,
    back: bool = False,
    xp: ModuleType = np,
) -> Array:
    """Rotate Ix/Iz components for each state along the effective field."""
    if xp is not np:
        return _tilt_functional(magnetization, tilts, back=back, xp=xp)
    for tilt in tilts:
        angle = -tilt.angle if back else tilt.angle
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        components = magnetization[..., [tilt.index_x, tilt.index_z], :]
        magnetization[..., [tilt.index_x, tilt.index_z], :] = (
            rotation_matrix @ components
        )
    return magnetization


def _tilt_functional(
    magnetization: Array,
    tilts: Iterable[EffectiveFieldTilt],
    *,
    back: bool,
    xp: ModuleType,
) -> Array:
    """Out-of-place version of the tilt for traced array namespaces."""
    magnetization = xp.asarray(magnetization)
    for tilt in tilts:
        angle = -tilt.angle if back else tilt.angle
        cos, sin = xp.cos(angle), xp.sin(angle)
        rotation_matrix = xp.stack([xp.stack([cos, -sin]), xp.stack([sin, cos])])
        indices = np.array([tilt.index_x, tilt.index_z])
        components = magnetization[..., indices, :]
        magnetization = magnetization.at[..., indices, :].set(
            rotation_matrix @ components
        )
    return magnetization
