# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""JAX twins of ChemEx's scientific constraint functions.

A twin reproduces the *maths* of a model-owned function (same formulas, same
evaluation order, same branch structure via ``jnp.where``) on traced values.
Value-domain checks (finiteness, sign, representability) are not repeated
here: they run on concrete values through ChemEx's own resolver
(:func:`chemex.parameters.program_evaluation.validate_concrete`).

Twins are keyed by the bound implementation (``inspect.unwrap`` of the
binder's callable), because function ids such as ``rates`` or
``populations`` name different functions in different models.  A function
without a twin raises :class:`MissingTwinError` instead of being attempted.

Branches on exact zeros (structurally absent exchange pathways) are evaluated
with both sides computed on safe operands; at such points the populations are
not differentiable and the twin returns the selected branch's derivative.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
from scipy import constants

from chemex.models import constraints as _constraints
from chemex.models.kinetic import _eyring
from chemex.models.kinetic import settings_2st_eyring as _eyring2
from chemex.models.kinetic import settings_3st_eyring as _eyring3
from chemex.models.kinetic import settings_4st_eyring as _eyring4
from chemex.models.kinetic import settings_nst as _nst
from chemex.nmr.rates import RatesIS

Twin = Callable[..., Any]


class MissingTwinError(NotImplementedError):
    """A scientific function has no JAX twin (yet)."""


def _safe(condition: Any, value: Any, fallback: float = 1.0) -> Any:
    """``value`` where ``condition`` is False, a harmless constant elsewhere."""
    return jnp.where(condition, fallback, value)


# ---------------------------------------------------------------------------
# Population constraints (chemex.models.constraints)
# ---------------------------------------------------------------------------
def pop_2st(kab: Any = 0.0, kba: Any = 0.0) -> dict[str, Any]:
    scale = jnp.maximum(jnp.abs(kab), jnp.abs(kba))
    zero = scale == 0.0
    scale = _safe(zero, scale)
    kab_scaled = kab / scale
    kba_scaled = kba / scale
    total = _safe(zero, kab_scaled + kba_scaled)
    return {
        "pa": jnp.where(zero, 1.0, kba_scaled / total),
        "pb": jnp.where(zero, 0.0, kab_scaled / total),
    }


def pop_3st(
    kab: Any = 0.0,
    kba: Any = 0.0,
    kac: Any = 0.0,
    kca: Any = 0.0,
    kbc: Any = 0.0,
    kcb: Any = 0.0,
) -> dict[str, Any]:
    no_a = (kab == 0.0) & (kba == 0.0) & (kac == 0.0) & (kca == 0.0)
    no_b = (kab == 0.0) & (kba == 0.0) & (kbc == 0.0) & (kcb == 0.0)
    no_c = (kac == 0.0) & (kca == 0.0) & (kbc == 0.0) & (kcb == 0.0)
    bc = pop_2st(kbc, kcb)
    ac = pop_2st(kac, kca)
    ab = pop_2st(kab, kba)

    scale = jnp.max(jnp.stack([kab, kba, kac, kca, kbc, kcb]))
    scale = _safe(scale == 0.0, scale)
    kab, kba, kac, kca, kbc, kcb = (r / scale for r in (kab, kba, kac, kca, kbc, kcb))
    wa = kba * kca + kba * kcb + kbc * kca
    wb = kab * kca + kab * kcb + kac * kcb
    wc = kac * kba + kac * kbc + kab * kbc
    total = wa + wb + wc
    empty = total == 0.0
    total = _safe(empty, total)
    general = {
        "pa": jnp.where(empty, 1.0, wa / total),
        "pb": jnp.where(empty, 0.0, wb / total),
        "pc": jnp.where(empty, 0.0, wc / total),
    }
    # Same precedence as the NumPy implementation: A absent, B absent, C absent.
    out = {}
    for name, a_absent, b_absent, c_absent in (
        ("pa", 0.0, ac["pa"], ab["pa"]),
        ("pb", bc["pa"], 0.0, ab["pb"]),
        ("pc", bc["pb"], ac["pb"], 0.0),
    ):
        value = jnp.where(no_c, c_absent, general[name])
        value = jnp.where(no_b, b_absent, value)
        out[name] = jnp.where(no_a, a_absent, value)
    return out


def _isclose_zero(value: Any) -> Any:
    # np.isclose(value, 0.0) with default tolerances: |value| <= 1e-8.
    return jnp.abs(value) <= 1e-8


def pop_4st(
    kab: Any = 0.0,
    kba: Any = 0.0,
    kac: Any = 0.0,
    kca: Any = 0.0,
    kad: Any = 0.0,
    kda: Any = 0.0,
    kbc: Any = 0.0,
    kcb: Any = 0.0,
    kbd: Any = 0.0,
    kdb: Any = 0.0,
    kcd: Any = 0.0,
    kdc: Any = 0.0,
) -> dict[str, Any]:
    no_a = _isclose_zero(kab + kba + kac + kca + kad + kda)
    no_b = _isclose_zero(kab + kba + kbc + kcb + kbd + kdb)
    no_c = _isclose_zero(kac + kca + kbc + kcb + kcd + kdc)
    no_d = _isclose_zero(kad + kda + kbd + kdb + kcd + kdc)
    without_a = pop_3st(kbc, kcb, kbd, kdb, kcd, kdc)
    without_b = pop_3st(kac, kca, kad, kda, kcd, kdc)
    without_c = pop_3st(kab, kba, kad, kda, kbd, kdb)
    without_d = pop_3st(kab, kba, kac, kca, kbc, kcb)

    one, zero = jnp.ones_like(kab + 0.0), jnp.zeros_like(kab + 0.0)
    mat = jnp.stack(
        [
            jnp.stack([-kab - kac - kad, kba + zero, kca + zero, kda + zero]),
            jnp.stack([kab + zero, -kba - kbc - kbd, kcb + zero, kdb + zero]),
            jnp.stack([kac + zero, kbc + zero, -kca - kcb - kcd, kdc + zero]),
            jnp.stack([one, one, one, one]),
        ]
    )
    singular = _isclose_zero(jnp.linalg.det(mat))
    safe_mat = jnp.where(singular, jnp.eye(4), mat)
    solution = jnp.linalg.solve(safe_mat, jnp.array([0.0, 0.0, 0.0, 1.0]))
    general = jnp.where(singular, jnp.array([1.0, 0.0, 0.0, 0.0]), solution)

    branches = (
        (no_a, (0.0, without_a["pa"], without_a["pb"], without_a["pc"])),
        (no_b, (without_b["pa"], 0.0, without_b["pb"], without_b["pc"])),
        (no_c, (without_c["pa"], without_c["pb"], 0.0, without_c["pc"])),
        (no_d, (without_d["pa"], without_d["pb"], without_d["pc"], 0.0)),
    )
    out = {}
    for index, name in enumerate(("pa", "pb", "pc", "pd")):
        value = general[index]
        for condition, values in reversed(branches):
            value = jnp.where(condition, values[index], value)
        out[name] = value
    return out


# ---------------------------------------------------------------------------
# Generic N-state models (settings_nst)
# ---------------------------------------------------------------------------
def population_complement(*independent_populations: Any) -> dict[str, Any]:
    total = independent_populations[0]
    for value in independent_populations[1:]:
        total = total + value
    return {"pa": 1.0 - total}


def pair_rates(kex: Any, p_i: Any, p_j: Any) -> dict[str, Any]:
    # ChemEx sorts (p_i, p_j) and computes the smaller directional rate first.
    # Select the operand explicitly instead of min/max: at p_i == p_j a
    # min/max derivative would split 50/50, while the rate is smooth there.
    degenerate = (kex == 0.0) | (p_i == 0.0) | (p_j == 0.0)
    denominator = _safe(degenerate, p_i + p_j)
    j_not_larger = p_j <= p_i
    smaller = jnp.where(j_not_larger, p_j, p_i)
    smaller_rate = kex * smaller / denominator
    larger_rate = kex - smaller_rate
    forward = jnp.where(j_not_larger, smaller_rate, larger_rate)
    reverse = jnp.where(j_not_larger, larger_rate, smaller_rate)
    forward = jnp.where(p_j == 0.0, 0.0, forward)
    reverse = jnp.where(p_j == 0.0, kex, reverse)
    forward = jnp.where(p_i == 0.0, kex, forward)
    reverse = jnp.where(p_i == 0.0, 0.0, reverse)
    forward = jnp.where(kex == 0.0, 0.0, forward)
    reverse = jnp.where(kex == 0.0, 0.0, reverse)
    return {"forward": forward, "reverse": reverse}


# ---------------------------------------------------------------------------
# Eyring models (_eyring, settings_{2,3,4}st_eyring)
# ---------------------------------------------------------------------------
_LOG_EYRING_FREQUENCY_FACTOR = math.log(constants.k / constants.h)


def _kelvin(temperature: Any) -> Any:
    return temperature + constants.zero_Celsius


def _directional_rate(
    initial: tuple[Any, Any], transition: tuple[Any, Any], temperature: Any
) -> Any:
    kelvin = _kelvin(temperature)
    activation_enthalpy = transition[0] - initial[0]
    activation_entropy = transition[1] - initial[1]
    return jnp.exp(
        _LOG_EYRING_FREQUENCY_FACTOR
        + jnp.log(kelvin)
        + activation_entropy / constants.R
        - activation_enthalpy / (constants.R * kelvin)
    )


def _thermodynamic_populations(
    states: dict[str, tuple[Any, Any]], temperature: Any
) -> dict[str, Any]:
    kelvin = _kelvin(temperature)
    log_weights = {
        name: entropy / constants.R - enthalpy / (constants.R * kelvin)
        for name, (enthalpy, entropy) in states.items()
    }
    reference = jnp.max(jnp.stack([jnp.asarray(v) for v in log_weights.values()]))
    relative = {name: value - reference for name, value in log_weights.items()}
    total = sum((jnp.exp(value) for value in relative.values()), start=0.0)
    log_total = jnp.log(total)
    return {name: jnp.exp(value - log_total) for name, value in relative.items()}


_REFERENCE = (0.0, 0.0)


def eyring_rate(
    initial_enthalpy: Any,
    initial_entropy: Any,
    transition_enthalpy: Any,
    transition_entropy: Any,
    temperature: Any,
) -> dict[str, Any]:
    return {
        "rate": _directional_rate(
            (initial_enthalpy, initial_entropy),
            (transition_enthalpy, transition_entropy),
            temperature,
        )
    }


def _edge_rates(
    states: dict[str, tuple[Any, Any]],
    transitions: dict[str, tuple[Any, Any]],
    temperature: Any,
) -> dict[str, Any]:
    rates = {}
    for edge, transition in transitions.items():
        for initial, final in (edge, edge[::-1]):
            rates[f"k{initial}{final}"] = _directional_rate(
                states[initial], transition, temperature
            )
    return rates


def kij_2st_eyring(dh_b, ds_b, dh_ab, ds_ab, temperature):
    return {
        "kab": _directional_rate(_REFERENCE, (dh_ab, ds_ab), temperature),
        "kba": _directional_rate((dh_b, ds_b), (dh_ab, ds_ab), temperature),
    }


def populations_2st_eyring(dh_b, ds_b, temperature):
    populations = _thermodynamic_populations(
        {"a": _REFERENCE, "b": (dh_b, ds_b)}, temperature
    )
    return {f"p{state}": value for state, value in populations.items()}


def kij_3st_eyring_linear(
    dh_b, ds_b, dh_c, ds_c, dh_ab, ds_ab, dh_bc, ds_bc, temperature
):
    states = {"a": _REFERENCE, "b": (dh_b, ds_b), "c": (dh_c, ds_c)}
    return _edge_rates(
        states, {"ab": (dh_ab, ds_ab), "bc": (dh_bc, ds_bc)}, temperature
    )


def kij_3st_eyring_fork(
    dh_b, ds_b, dh_c, ds_c, dh_ab, ds_ab, dh_ac, ds_ac, temperature
):
    states = {"a": _REFERENCE, "b": (dh_b, ds_b), "c": (dh_c, ds_c)}
    return _edge_rates(
        states, {"ab": (dh_ab, ds_ab), "ac": (dh_ac, ds_ac)}, temperature
    )


def populations_3st_eyring(dh_b, ds_b, dh_c, ds_c, temperature):
    populations = _thermodynamic_populations(
        {"a": _REFERENCE, "b": (dh_b, ds_b), "c": (dh_c, ds_c)}, temperature
    )
    return {f"p{state}": value for state, value in populations.items()}


def kij_4st_eyring(*args):
    dh_b, ds_b, dh_c, ds_c, dh_d, ds_d = args[:6]
    edges = args[6:18]
    temperature = args[18]
    states = {"a": _REFERENCE, "b": (dh_b, ds_b), "c": (dh_c, ds_c), "d": (dh_d, ds_d)}
    transitions = {
        edge: (edges[2 * i], edges[2 * i + 1]) for i, edge in enumerate(_eyring4.EDGES)
    }
    return _edge_rates(states, transitions, temperature)


def populations_4st_eyring(dh_b, ds_b, dh_c, ds_c, dh_d, ds_d, temperature):
    populations = _thermodynamic_populations(
        {"a": _REFERENCE, "b": (dh_b, ds_b), "c": (dh_c, ds_c), "d": (dh_d, ds_d)},
        temperature,
    )
    return {f"p{state}": value for state, value in populations.items()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TWINS: dict[Callable[..., Any], Twin] = {
    inspect.unwrap(_constraints.pop_2st): pop_2st,
    inspect.unwrap(_constraints.pop_3st): pop_3st,
    inspect.unwrap(_constraints.pop_4st): pop_4st,
    _nst.calculate_population_complement: population_complement,
    _nst.calculate_pair_rates: pair_rates,
    _eyring.calculate_rate_component: eyring_rate,
    inspect.unwrap(_eyring2.calculate_kij_2st_eyring): kij_2st_eyring,
    inspect.unwrap(_eyring2.calculate_populations_2st_eyring): populations_2st_eyring,
    inspect.unwrap(_eyring3.calculate_kij_3st_eyring_linear): kij_3st_eyring_linear,
    inspect.unwrap(_eyring3.calculate_kij_3st_eyring_fork): kij_3st_eyring_fork,
    inspect.unwrap(_eyring3.calculate_populations_3st_eyring): populations_3st_eyring,
    inspect.unwrap(_eyring4.calculate_kij_4st_eyring): kij_4st_eyring,
    inspect.unwrap(_eyring4.calculate_populations_4st_eyring): populations_4st_eyring,
}


def _all_twins() -> dict[Callable[..., Any], Twin]:
    from chemex.backend import jax_binding, jax_oligomerization

    return TWINS | jax_binding.TWINS | jax_oligomerization.TWINS


_REGISTRY: dict[Callable[..., Any], Twin] | None = None


def twin_for(function: Callable[..., Any]) -> Twin | None:
    """The JAX twin of a bound scientific function, if any."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _all_twins()
    return _REGISTRY.get(inspect.unwrap(function))


def _maximum(*args: Any) -> Any:
    result = args[0]
    for value in args[1:]:
        result = jnp.maximum(result, value)
    return result


def evaluate(
    function_id: str, function: Callable[..., Any], args: Sequence[Any]
) -> Any:
    """Evaluate a bound scientific function on (possibly traced) arguments."""
    if function is max:
        return _maximum(*args)
    if isinstance(function, RatesIS):
        # Model-free rates are plain arithmetic in tauc/s2/khh; h_frq is an
        # expression literal, so the NumPy constants it feeds stay concrete.
        return function(*args)
    twin = twin_for(function)
    if twin is None:
        qualname = getattr(function, "__qualname__", type(function).__qualname__)
        msg = (
            f"Scientific function {function_id!r} "
            f"({getattr(function, '__module__', '?')}.{qualname}) has no JAX twin"
        )
        raise MissingTwinError(msg)
    return twin(*args)
