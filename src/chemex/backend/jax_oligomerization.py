# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""JAX twins of the oligomerization models (``models/kinetic/_oligomerization.py``).

ChemEx solves the normalized mass balance for the log monomer fraction ``y``

    g(y) = logsumexp(y, log n_k + c_k + n_k y) = 0

with ``scipy.optimize.brentq``.  ``g`` is increasing and convex in ``y`` and
``g(0) >= 0``, so Newton's method started at ``y = 0`` converges monotonically
(every iterate stays at or above the root) and quadratically; the twin runs it
in a ``lax.while_loop`` until the step is at rounding level.  The solve is a
``jax.custom_jvp`` whose tangent comes from the implicit function theorem,
``dy = -(dg/dc . dc) / (dg/dy)``, so derivatives never differentiate through
the iteration.  Everything after the root is closed-form and mirrors ChemEx.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from chemex.models.kinetic import _oligomerization as _olig
from chemex.models.kinetic import settings_2st_monomer_dimer as _md
from chemex.models.kinetic import settings_2st_monomer_tetramer as _mte
from chemex.models.kinetic import settings_2st_monomer_trimer as _mtr
from chemex.models.kinetic import settings_3st_monomer_dimer_tetramer as _mdte
from chemex.models.kinetic import settings_3st_monomer_dimer_trimer as _mdtr
from chemex.models.kinetic._oligomerization import OligomerizationEquilibrium

_EPS = float(np.finfo(np.float64).eps)
_MAX_ITERATIONS = 200


def _logsumexp(stacked: jax.Array) -> jax.Array:
    maximum = jnp.max(stacked)
    return maximum + jnp.log(jnp.sum(jnp.exp(stacked - maximum)))


def _terms(
    stoichiometries: tuple[int, ...], log_coefficients: jax.Array, y: Any
) -> Any:
    n = jnp.asarray(stoichiometries, dtype=jnp.float64)
    return jnp.concatenate(
        [jnp.reshape(y, (1,)), jnp.log(n) + log_coefficients + n * y]
    )


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def log_monomer_fraction(
    stoichiometries: tuple[int, ...], log_coefficients: jax.Array
) -> jax.Array:
    """Root ``y <= 0`` of the normalized log mass balance (Newton from 0)."""
    n = jnp.asarray(stoichiometries, dtype=jnp.float64)

    def step(y: jax.Array) -> jax.Array:
        terms = _terms(stoichiometries, log_coefficients, y)
        weights = jax.nn.softmax(terms)
        slope = weights[0] + jnp.sum(weights[1:] * n)
        return _logsumexp(terms) / slope

    def cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        y, delta, iteration = state
        converged = jnp.abs(delta) <= 2.0 * _EPS * jnp.maximum(1.0, jnp.abs(y))
        return (iteration < _MAX_ITERATIONS) & ~converged

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        y, _, iteration = state
        delta = step(y)
        return y - delta, delta, iteration + 1

    zero = jnp.zeros((), dtype=jnp.float64)
    state = (zero, jnp.asarray(jnp.inf), jnp.asarray(0))
    y, _, _ = lax.while_loop(cond, body, state)
    return y


@log_monomer_fraction.defjvp
def _log_monomer_fraction_jvp(
    stoichiometries: tuple[int, ...],
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    (log_coefficients,) = primals
    (d_log_coefficients,) = tangents
    y = log_monomer_fraction(stoichiometries, log_coefficients)
    n = jnp.asarray(stoichiometries, dtype=jnp.float64)
    weights = jax.nn.softmax(_terms(stoichiometries, log_coefficients, y))
    dg_dy = weights[0] + jnp.sum(weights[1:] * n)
    dg_dc = weights[1:]
    return y, -jnp.dot(dg_dc, d_log_coefficients) / dg_dy


def log_equilibrium_coefficient(p_total: Any, p_total_power: int, *kds: Any) -> Any:
    total = jnp.log(kds[0])
    for kd in kds[1:]:
        total = total + jnp.log(kd)
    return p_total_power * jnp.log(p_total) - total


def solve_oligomerization_fractions(
    terms: tuple[tuple[int, Any], ...],
) -> OligomerizationEquilibrium:
    stoichiometries = tuple(int(s) for s, _ in terms)
    log_coefficients = jnp.stack([jnp.asarray(c, dtype=jnp.float64) for _, c in terms])
    y = log_monomer_fraction(stoichiometries, log_coefficients)
    n = jnp.asarray(stoichiometries, dtype=jnp.float64)
    raw_log_oligomer = log_coefficients + n * y
    raw_log_tagged = jnp.concatenate(
        [jnp.reshape(y, (1,)), jnp.log(n) + raw_log_oligomer]
    )
    log_tagged = raw_log_tagged - _logsumexp(raw_log_tagged)
    tagged = jnp.exp(log_tagged)
    tagged = tagged + (1.0 - jnp.sum(tagged)) * (
        jnp.arange(tagged.shape[0]) == jnp.argmax(tagged)
    )
    count = len(stoichiometries)
    # Same record ChemEx returns, holding (possibly traced) JAX arrays.
    fields: dict[str, Any] = {
        "monomer_fraction": tagged[0],
        "oligomer_fractions": tuple(tagged[k + 1] / n[k] for k in range(count)),
        "log_monomer_fraction": log_tagged[0],
        "log_oligomer_fractions": tuple(
            log_tagged[k + 1] - jnp.log(n[k]) for k in range(count)
        ),
        "log_tagged_fractions": tuple(log_tagged[k] for k in range(count + 1)),
    }
    return OligomerizationEquilibrium(**fields)


def concentrations_from_log_fractions(
    p_total: Any, species: tuple[tuple[int, Any], ...]
) -> tuple[Any, ...]:
    log_p = jnp.log(p_total)
    log_concentrations = jnp.stack([log_p + lf for _, lf in species])
    underflow = log_concentrations < _olig.LOG_MIN_POSITIVE_FLOAT
    concentrations = jnp.where(underflow, 0.0, jnp.exp(log_concentrations))
    stoichiometries = jnp.asarray([s for s, _ in species], dtype=jnp.float64)
    tagged_mass = stoichiometries * (concentrations / p_total)
    correction = 1.0 - jnp.sum(tagged_mass)
    largest = jnp.argmax(tagged_mass)
    one_hot = jnp.arange(len(species)) == largest
    concentrations = (
        concentrations + one_hot * correction * p_total / stoichiometries[largest]
    )
    return tuple(concentrations[i] for i in range(len(species)))


def scale_reversible_rate(rate: Any, factor: float) -> Any:
    return jnp.where(rate == 0.0, 0.0, rate * factor)


def detailed_balance_forward_rate(
    reverse_rate: Any, log_source_fraction: Any, log_destination_fraction: Any
) -> Any:
    zero = reverse_rate == 0.0
    log_rate = (
        jnp.log(jnp.where(zero, 1.0, reverse_rate))
        + log_destination_fraction
        - log_source_fraction
    )
    return jnp.where(zero, 0.0, jnp.exp(log_rate))


# --- model glue -------------------------------------------------------------
@dataclass(frozen=True)
class _Model:
    """Oligomer terms: (stoichiometry, p_total power, indices into the KDs)."""

    terms: tuple[tuple[int, int, tuple[int, ...]], ...]
    names: tuple[str, ...]

    def equilibrium(
        self, p_total: Any, kds: tuple[Any, ...]
    ) -> OligomerizationEquilibrium:
        return solve_oligomerization_fractions(
            tuple(
                (s, log_equilibrium_coefficient(p_total, power, *(kds[i] for i in idx)))
                for s, power, idx in self.terms
            )
        )


def _zero_total(p_total: Any) -> bool:
    if isinstance(p_total, Real | np.number):
        return float(p_total) == 0.0
    msg = "oligomerization twins need a concrete P_total (a condition value)"
    raise TypeError(msg)


def _concentrations(model: _Model, p_total: Any, *kds: Any) -> dict[str, Any]:
    if _zero_total(p_total):
        return {"monomer": 0.0} | dict.fromkeys(model.names, 0.0)
    eq = model.equilibrium(p_total, kds)
    values = concentrations_from_log_fractions(
        p_total,
        (
            (1, eq.log_monomer_fraction),
            *(
                (s, eq.log_oligomer_fractions[k])
                for k, (s, _, _) in enumerate(model.terms)
            ),
        ),
    )
    return {"monomer": values[0]} | dict(zip(model.names, values[1:], strict=True))


def _populations(model: _Model, p_total: Any, *kds: Any) -> dict[str, Any]:
    keys = ("pa", "pb", "pc")[: len(model.terms) + 1]
    if _zero_total(p_total):
        return dict(zip(keys, (1.0, *([0.0] * len(model.terms))), strict=True))
    eq = model.equilibrium(p_total, kds)
    values = (
        eq.monomer_fraction,
        *(
            float(s) * eq.oligomer_fractions[k]
            for k, (s, _, _) in enumerate(model.terms)
        ),
    )
    return dict(zip(keys, values, strict=True))


_DIMER = _Model(((2, 1, (0,)),), ("dimer",))
_TRIMER = _Model(((3, 2, (0,)),), ("trimer",))
_TETRAMER = _Model(((4, 3, (0,)),), ("tetramer",))
_DIMER_TRIMER = _Model(((2, 1, (0,)), (3, 2, (0, 1))), ("dimer", "trimer"))
_DIMER_TETRAMER = _Model(((2, 1, (0,)), (4, 3, (0, 0, 1))), ("dimer", "tetramer"))


def _two_state_rates(model: _Model) -> Callable[..., dict[str, Any]]:
    def rates(p_total: Any, kd: Any, koff: Any) -> dict[str, Any]:
        if _zero_total(p_total):
            return {"kab": 0.0}
        log_tagged = model.equilibrium(p_total, (kd,)).log_tagged_fractions
        return {
            "kab": detailed_balance_forward_rate(koff, log_tagged[0], log_tagged[1])
        }

    return rates


def dimer_trimer_rates(
    p_total: Any, kd1: Any, kd2: Any, koff1: Any, koff2: Any
) -> dict[str, Any]:
    kca = scale_reversible_rate(koff2, 1.0 / 3.0)
    kcb = scale_reversible_rate(koff2, 2.0 / 3.0)
    if _zero_total(p_total):
        return {"kab": 0.0, "kac": 0.0, "kca": kca, "kbc": 0.0, "kcb": kcb}
    log_pa, log_pb, log_pc = _DIMER_TRIMER.equilibrium(
        p_total, (kd1, kd2)
    ).log_tagged_fractions
    return {
        "kab": detailed_balance_forward_rate(koff1, log_pa, log_pb),
        "kac": detailed_balance_forward_rate(kca, log_pa, log_pc),
        "kca": kca,
        "kbc": detailed_balance_forward_rate(kcb, log_pb, log_pc),
        "kcb": kcb,
    }


def dimer_tetramer_rates(
    p_total: Any, kd1: Any, kd2: Any, koff1: Any, koff2: Any
) -> dict[str, Any]:
    if _zero_total(p_total):
        return {"kab": 0.0, "kbc": 0.0}
    log_pa, log_pb, log_pc = _DIMER_TETRAMER.equilibrium(
        p_total, (kd1, kd2)
    ).log_tagged_fractions
    return {
        "kab": detailed_balance_forward_rate(koff1, log_pa, log_pb),
        "kbc": detailed_balance_forward_rate(koff2, log_pb, log_pc),
    }


def _key(function: Callable[..., Any]) -> Callable[..., Any]:
    return inspect.unwrap(function)


TWINS: dict[Callable[..., Any], Callable[..., Any]] = {}
for _module, _model in ((_md, _DIMER), (_mtr, _TRIMER), (_mte, _TETRAMER)):
    TWINS[_key(_module.calculate_concentrations)] = partial(_concentrations, _model)
    TWINS[_key(_module.calculate_populations)] = partial(_populations, _model)
    TWINS[_key(_module.calculate_rates)] = _two_state_rates(_model)
for _module, _model, _rates in (
    (_mdtr, _DIMER_TRIMER, dimer_trimer_rates),
    (_mdte, _DIMER_TETRAMER, dimer_tetramer_rates),
):
    TWINS[_key(_module.calculate_concentrations)] = partial(_concentrations, _model)
    TWINS[_key(_module.calculate_populations)] = partial(_populations, _model)
    TWINS[_key(_module.calculate_rates)] = _rates
