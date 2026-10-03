# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""JAX twins of the finite-pool binding models (``models/kinetic/_binding.py``).

Each twin mirrors the corresponding NumPy function line by line: same
log-domain formulas, same occupancy-limit switch, same mass-balance
correction applied to the largest species.  Branches that depend only on the
condition totals (``l_total == 0``, ``p_total == l_total``, ``p <= l``) are
Python branches when those totals are concrete numbers, exactly as upstream;
parameter-dependent branches use ``jnp.where`` with safe operands on the
branch not taken, so no NaN reaches the selected value or its derivative.
``math.fsum`` is replaced by an ordinary sum (differences at the ulp level).

Validation (finiteness, sign, representability, mass balance, association
law) is left to ChemEx's resolver on concrete values.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

import jax.numpy as jnp
import numpy as np

from chemex.models.kinetic import _binding
from chemex.models.kinetic import settings_2st_binding as _b2
from chemex.models.kinetic import settings_3st_binding_2st_partner as _b3p
from chemex.models.kinetic import settings_3st_binding_cs as _bcs
from chemex.models.kinetic import settings_3st_binding_if as _bif
from chemex.models.kinetic import settings_3st_double_binding as _b3d
from chemex.models.kinetic import settings_4st_binding_2st_partner as _b4p
from chemex.models.kinetic import settings_4st_binding_3_bound_states as _b43
from chemex.models.kinetic._binding import BindingEquilibrium

_LOG_TWO = math.log(2.0)
_LOG_FOUR = math.log(4.0)
_LOG_MIN_POSITIVE_FLOAT = _binding._LOG_MIN_POSITIVE_FLOAT
_LOG_MAX_FLOAT = _binding._LOG_MAX_FLOAT
NEG_INF = -jnp.inf


def _concrete(*values: Any) -> bool:
    return all(isinstance(v, Real | np.number) for v in values)


def _stack(values: Sequence[Any]) -> Any:
    return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])


# --- _binding primitives ----------------------------------------------------
def logsumexp(values: Sequence[Any]) -> Any:
    stacked = _stack(values)
    maximum = jnp.max(stacked)
    empty = maximum == -jnp.inf
    safe_max = jnp.where(empty, 0.0, maximum)
    total = jnp.sum(jnp.exp(stacked - safe_max))
    return jnp.where(empty, -jnp.inf, safe_max + jnp.log(jnp.where(empty, 1.0, total)))


def _with_correction(values: Any, target: Any) -> Any:
    """Add ``target - sum(values)`` to the (first) largest entry."""
    correction = target - jnp.sum(values)
    return values + correction * (jnp.arange(values.shape[0]) == jnp.argmax(values))


def normalized_weights(log_weights: Sequence[Any]) -> tuple[Any, ...]:
    stacked = _stack(log_weights)
    maximum = jnp.max(stacked)
    removed = stacked == -jnp.inf
    raw = jnp.where(removed, 0.0, jnp.exp(jnp.where(removed, 0.0, stacked - maximum)))
    weights = _with_correction(raw / jnp.sum(raw), 1.0)
    return tuple(weights[i] for i in range(weights.shape[0]))


def normalized_log_weights(log_weights: Sequence[Any]) -> tuple[Any, ...]:
    log_total = logsumexp(log_weights)
    return tuple(jnp.where(w == -jnp.inf, -jnp.inf, w - log_total) for w in log_weights)


def distribute(total: Any, fractions: Sequence[Any]) -> tuple[Any, ...]:
    concentrations = _with_correction(total * _stack(fractions), total)
    concentrations = jnp.where(total == 0.0, 0.0, concentrations)
    return tuple(concentrations[i] for i in range(concentrations.shape[0]))


def _equal_pool_free(total: Any, log_apparent_kd: Any) -> Any:
    in_range = (log_apparent_kd >= _LOG_MIN_POSITIVE_FLOAT) & (
        log_apparent_kd <= _LOG_MAX_FLOAT
    )
    log_free = 0.5 * (log_apparent_kd + jnp.log(total))
    safe_log_kd = jnp.where(in_range, log_apparent_kd, 0.0)
    kd = jnp.exp(safe_log_kd)
    scale = jnp.maximum(kd, total)
    sqrt_kd = jnp.sqrt(kd / scale)
    sqrt_sum = jnp.sqrt(kd / scale + 4.0 * (total / scale))
    fraction = 2.0 * sqrt_kd / (sqrt_kd + sqrt_sum)
    return jnp.where(in_range, total * fraction, jnp.exp(log_free))


def finite_pool_totals(p_total: Any, l_total: Any, log_apparent_kd: Any) -> tuple:
    log_protein = jnp.log(p_total)
    log_ligand = jnp.log(l_total)
    log_sum = logsumexp((log_protein, log_ligand, log_apparent_kd))
    difference = abs(p_total - l_total)
    if _concrete(difference):
        equal = difference == 0.0
        log_difference = -math.inf if equal else math.log(difference)
    else:
        equal = difference == 0.0
        log_difference = jnp.where(
            equal, -jnp.inf, jnp.log(jnp.where(equal, 1.0, difference))
        )
    log_discriminant = logsumexp(
        (
            2.0 * log_difference,
            _LOG_TWO + log_apparent_kd + logsumexp((log_protein, log_ligand)),
            2.0 * log_apparent_kd,
        )
    )
    log_sqrt_discriminant = 0.5 * log_discriminant
    log_bound = (
        _LOG_TWO
        + log_protein
        + log_ligand
        - logsumexp((log_sum, log_sqrt_discriminant))
    )
    limiting = (
        jnp.minimum(p_total, l_total)
        if not _concrete(p_total, l_total)
        else min(p_total, l_total)
    )
    log_limiting = jnp.log(limiting)
    weak = log_bound - log_limiting <= -_LOG_TWO

    # Weakly occupied limit (selected where ``weak``); safe operands elsewhere.
    bound_weak = jnp.exp(log_bound)
    protein_free_weak = p_total - bound_weak
    ligand_free_weak = l_total - bound_weak
    log_protein_free_weak = jnp.log(jnp.where(weak, protein_free_weak, 1.0))
    log_ligand_free_weak = jnp.log(jnp.where(weak, ligand_free_weak, 1.0))

    # Strongly occupied limit.
    log_linear = logsumexp((log_difference, log_apparent_kd))
    log_sqrt = 0.5 * logsumexp(
        (2.0 * log_linear, _LOG_FOUR + log_apparent_kd + log_limiting)
    )
    log_free_limiting = (
        _LOG_TWO + log_apparent_kd + log_limiting - logsumexp((log_linear, log_sqrt))
    )
    if _concrete(difference):
        if equal:
            free_limiting = _equal_pool_free(limiting, log_apparent_kd)
            log_free_limiting = jnp.where(
                free_limiting > 0.0,
                jnp.log(jnp.where(free_limiting > 0.0, free_limiting, 1.0)),
                log_free_limiting,
            )
        else:
            free_limiting = jnp.exp(log_free_limiting)
    else:
        equal_free = _equal_pool_free(limiting, log_apparent_kd)
        free_limiting = jnp.where(equal, equal_free, jnp.exp(log_free_limiting))
        log_free_limiting = jnp.where(
            equal & (free_limiting > 0.0),
            jnp.log(jnp.where(free_limiting > 0.0, free_limiting, 1.0)),
            log_free_limiting,
        )
    bound_strong = limiting - free_limiting
    log_excess_free = logsumexp((log_difference, log_free_limiting))
    excess_free = difference + free_limiting
    protein_limited = p_total <= l_total
    if _concrete(p_total, l_total):
        if protein_limited:
            strong = (free_limiting, excess_free, log_free_limiting, log_excess_free)
        else:
            strong = (excess_free, free_limiting, log_excess_free, log_free_limiting)
    else:
        strong = (
            jnp.where(protein_limited, free_limiting, excess_free),
            jnp.where(protein_limited, excess_free, free_limiting),
            jnp.where(protein_limited, log_free_limiting, log_excess_free),
            jnp.where(protein_limited, log_excess_free, log_free_limiting),
        )
    protein_free = jnp.where(weak, protein_free_weak, strong[0])
    ligand_free = jnp.where(weak, ligand_free_weak, strong[1])
    bound = jnp.where(weak, bound_weak, bound_strong)
    log_protein_free = jnp.where(weak, log_protein_free_weak, strong[2])
    log_ligand_free = jnp.where(weak, log_ligand_free_weak, strong[3])
    return (
        protein_free,
        ligand_free,
        bound,
        log_protein_free,
        log_ligand_free,
        log_bound,
    )


def solve_binding_equilibrium(
    p_total: Any,
    l_total: Any,
    *,
    free_protein_log_weights: tuple[Any, ...] = (0.0,),
    free_ligand_log_weights: tuple[Any, ...],
    complex_log_weights: tuple[Any, ...],
    edge_log_ratios: tuple[Any, ...] = (),
) -> BindingEquilibrium:
    protein_fractions = normalized_weights(free_protein_log_weights)
    free_fractions = normalized_weights(free_ligand_log_weights)
    complex_fractions = normalized_weights(complex_log_weights)
    protein_log_fractions = normalized_log_weights(free_protein_log_weights)
    free_log_fractions = normalized_log_weights(free_ligand_log_weights)
    complex_log_fractions = normalized_log_weights(complex_log_weights)
    log_protein_weight = logsumexp(free_protein_log_weights)
    log_free_weight = logsumexp(free_ligand_log_weights)
    log_complex_weight = logsumexp(complex_log_weights)
    log_apparent_kd = log_protein_weight + log_free_weight - log_complex_weight

    if _concrete(l_total) and l_total == 0.0:
        protein_free, ligand_free, bound_total = p_total, 0.0, 0.0
        log_protein_free = jnp.log(p_total)
        log_bound = -jnp.inf  # (log free ligand only feeds validation)
    elif _concrete(l_total):
        (
            protein_free,
            ligand_free,
            bound_total,
            log_protein_free,
            _log_ligand_free,
            log_bound,
        ) = finite_pool_totals(p_total, l_total, log_apparent_kd)
    else:
        no_ligand = l_total == 0.0
        totals = finite_pool_totals(
            p_total, jnp.where(no_ligand, 1.0, l_total), log_apparent_kd
        )
        protein_free = jnp.where(no_ligand, p_total, totals[0])
        ligand_free = jnp.where(no_ligand, 0.0, totals[1])
        bound_total = jnp.where(no_ligand, 0.0, totals[2])
        log_protein_free = jnp.where(no_ligand, jnp.log(p_total), totals[3])
        log_bound = jnp.where(no_ligand, -jnp.inf, totals[5])

    log_p_total = jnp.log(p_total)
    raw_log_populations = (
        *(
            jnp.where(
                w == -jnp.inf,
                -jnp.inf,
                log_protein_free + w - log_protein_weight - log_p_total,
            )
            for w in free_protein_log_weights
        ),
        *(
            jnp.where(
                w == -jnp.inf,
                -jnp.inf,
                log_bound + w - log_complex_weight - log_p_total,
            )
            for w in complex_log_weights
        ),
    )
    populations = normalized_weights(raw_log_populations)
    log_population_total = logsumexp(raw_log_populations)
    log_populations = tuple(
        jnp.where(p == -jnp.inf, -jnp.inf, p - log_population_total)
        for p in raw_log_populations
    )
    # Same record ChemEx returns, holding (possibly traced) JAX arrays.
    fields: dict[str, Any] = {
        "protein_free": protein_free,
        "ligand_free": ligand_free,
        "bound_total": bound_total,
        "free_proteins": distribute(protein_free, protein_fractions),
        "free_ligands": distribute(ligand_free, free_fractions),
        "complexes": distribute(bound_total, complex_fractions),
        "populations": populations,
        "log_populations": log_populations,
        "free_protein_log_fractions": protein_log_fractions,
        "free_ligand_log_fractions": free_log_fractions,
        "complex_log_fractions": complex_log_fractions,
        "edge_log_ratios": tuple(edge_log_ratios),
        "log_apparent_kd": log_apparent_kd,
    }
    return BindingEquilibrium(**fields)


def detailed_balance_rate(
    reverse_rate: Any, log_source_population: Any, log_destination_population: Any
) -> Any:
    zero = (reverse_rate == 0.0) | (log_destination_population == -jnp.inf)
    log_rate = (
        jnp.log(jnp.where(zero, 1.0, reverse_rate))
        + jnp.where(zero, 0.0, log_destination_population)
        - jnp.where(zero, 0.0, log_source_population)
    )
    return jnp.where(zero, 0.0, jnp.exp(log_rate))


def split_exchange_rate(
    total_rate: Any, log_source_weight: Any, log_destination_weight: Any
) -> tuple[Any, Any]:
    zero_total = total_rate == 0.0
    log_source_fraction, log_destination_fraction = normalized_log_weights(
        (log_source_weight, log_destination_weight)
    )
    log_total_rate = jnp.log(jnp.where(zero_total, 1.0, total_rate))
    forward = jnp.where(
        log_destination_fraction == -jnp.inf,
        0.0,
        jnp.exp(log_total_rate + log_destination_fraction),
    )
    reverse = jnp.where(
        log_source_fraction == -jnp.inf,
        0.0,
        jnp.exp(log_total_rate + log_source_fraction),
    )
    rates = _with_correction(jnp.stack([forward, reverse]), total_rate)
    return jnp.where(zero_total, 0.0, rates[0]), jnp.where(zero_total, 0.0, rates[1])


def log_binding_weight(kd: Any) -> Any:
    return -jnp.log(kd)


def log_equilibrium_ratio(ratio: Any) -> Any:
    zero = ratio == 0.0
    return jnp.where(zero, -jnp.inf, jnp.log(jnp.where(zero, 1.0, ratio)))


def report_only_positive_log_value(log_value: Any) -> Any:
    invalid = jnp.isnan(log_value) | (log_value < _LOG_MIN_POSITIVE_FLOAT)
    overflow = log_value > _LOG_MAX_FLOAT
    safe = jnp.where(invalid | overflow, 0.0, log_value)
    return jnp.where(invalid, jnp.nan, jnp.where(overflow, jnp.inf, jnp.exp(safe)))


# --- settings_2st_binding ---------------------------------------------------
def _eq_2st(p_total, l_total, kd):
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_ligand_log_weights=(0.0,),
        complex_log_weights=(log_binding_weight(kd),),
    )


def b2_concentrations(p_total, l_total, kd):
    eq = _eq_2st(p_total, l_total, kd)
    return {"p_free": eq.protein_free, "l_free": eq.ligand_free, "pl": eq.complexes[0]}


def b2_rates(p_total, l_total, kd, koff):
    eq = _eq_2st(p_total, l_total, kd)
    return {
        "kab": detailed_balance_rate(koff, eq.log_populations[0], eq.log_populations[1])
    }


def b2_populations(p_total, l_total, kd):
    pa, pb = _eq_2st(p_total, l_total, kd).populations
    return {"pa": pa, "pb": pb}


# --- settings_3st_binding_2st_partner ---------------------------------------
def _eq_3p(p_total, l_total, kd1, kd2, keq):
    log_keq = log_equilibrium_ratio(keq)
    log_pl1 = log_binding_weight(kd1)
    log_pl2 = log_keq + log_binding_weight(kd2)
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_ligand_log_weights=(0.0, log_keq),
        complex_log_weights=(log_pl1, log_pl2),
        edge_log_ratios=(log_pl2 - log_pl1,),
    )


def b3p_concentrations(p_total, l_total, kd1, kd2, keq):
    eq = _eq_3p(p_total, l_total, kd1, kd2, keq)
    l1, l2 = eq.free_ligands
    pl1, pl2 = eq.complexes
    return {"p": eq.protein_free, "l1": l1, "l2": l2, "pl1": pl1, "pl2": pl2}


def b3p_populations(p_total, l_total, kd1, kd2, keq):
    pa, pb, pc = _eq_3p(p_total, l_total, kd1, kd2, keq).populations
    return {"pa": pa, "pb": pb, "pc": pc}


def b3p_rates(p_total, l_total, kd1, kd2, keq, koff_ab, koff_ac, kex_bc):
    eq = _eq_3p(p_total, l_total, kd1, kd2, keq)
    log_pa, log_pb, log_pc = eq.log_populations
    kbc, kcb = split_exchange_rate(kex_bc, 0.0, eq.edge_log_ratios[0])
    return {
        "kab": detailed_balance_rate(koff_ab, log_pa, log_pb),
        "kac": detailed_balance_rate(koff_ac, log_pa, log_pc),
        "kbc": kbc,
        "kcb": kcb,
    }


# --- settings_3st_double_binding --------------------------------------------
def _eq_3d(p_total, l_total, kd_ab, kd_ac):
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_ligand_log_weights=(0.0,),
        complex_log_weights=(log_binding_weight(kd_ab), log_binding_weight(kd_ac)),
    )


def b3d_concentrations(p_total, l_total, kd_ab, kd_ac):
    eq = _eq_3d(p_total, l_total, kd_ab, kd_ac)
    return {
        "pfree": eq.protein_free,
        "lfree": eq.ligand_free,
        "pl1": eq.complexes[0],
        "pl2": eq.complexes[1],
    }


def b3d_populations(p_total, l_total, kd_ab, kd_ac):
    pa, pb, pc = _eq_3d(p_total, l_total, kd_ab, kd_ac).populations
    return {"pa": pa, "pb": pb, "pc": pc}


def b3d_rates(p_total, l_total, kd_ab, kd_ac, koff_ab, koff_ac):
    eq = _eq_3d(p_total, l_total, kd_ab, kd_ac)
    log_pa, log_pb, log_pc = eq.log_populations
    return {
        "kab": detailed_balance_rate(koff_ab, log_pa, log_pb),
        "kac": detailed_balance_rate(koff_ac, log_pa, log_pc),
    }


# --- settings_4st_binding_2st_partner ---------------------------------------
def _eq_4p(p_total, l_total, kd1, kd2, keq_l, keq_pl):
    log_keq_l = log_equilibrium_ratio(keq_l)
    log_keq_pl = log_equilibrium_ratio(keq_pl)
    log_pl1 = log_binding_weight(kd1)
    log_pl2 = log_keq_l + log_binding_weight(kd2)
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_ligand_log_weights=(0.0, log_keq_l),
        complex_log_weights=(log_pl1, log_pl2, log_pl2 + log_keq_pl),
        edge_log_ratios=(log_pl2 - log_pl1, log_keq_pl),
    )


def b4p_concentrations(p_total, l_total, kd1, kd2, keq_l, keq_pl):
    eq = _eq_4p(p_total, l_total, kd1, kd2, keq_l, keq_pl)
    l1, l2 = eq.free_ligands
    pl1, pl2, pl3 = eq.complexes
    return {
        "p": eq.protein_free,
        "l1": l1,
        "l2": l2,
        "pl1": pl1,
        "pl2": pl2,
        "pl3": pl3,
    }


def b4p_rates(
    p_total, l_total, kd1, kd2, keq_l, keq_pl, koff_ab, koff_ac, kex_bc, kex_cd
):
    eq = _eq_4p(p_total, l_total, kd1, kd2, keq_l, keq_pl)
    log_pa, log_pb, log_pc, _log_pd = eq.log_populations
    kbc, kcb = split_exchange_rate(kex_bc, 0.0, eq.edge_log_ratios[0])
    kcd, kdc = split_exchange_rate(kex_cd, 0.0, eq.edge_log_ratios[1])
    return {
        "kab": detailed_balance_rate(koff_ab, log_pa, log_pb),
        "kac": detailed_balance_rate(koff_ac, log_pa, log_pc),
        "kbc": kbc,
        "kcb": kcb,
        "kcd": kcd,
        "kdc": kdc,
    }


def b4p_populations(p_total, l_total, kd1, kd2, keq_l, keq_pl):
    pa, pb, pc, pd = _eq_4p(p_total, l_total, kd1, kd2, keq_l, keq_pl).populations
    return {"pa": pa, "pb": pb, "pc": pc, "pd": pd}


# --- settings_4st_binding_3_bound_states ------------------------------------
def _eq_43(p_total, l_total, kd_ab, keq_bc, keq_cd):
    log_keq_bc = log_equilibrium_ratio(keq_bc)
    log_keq_cd = log_equilibrium_ratio(keq_cd)
    log_pl1 = log_binding_weight(kd_ab)
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_ligand_log_weights=(0.0,),
        complex_log_weights=(
            log_pl1,
            log_pl1 + log_keq_bc,
            log_pl1 + log_keq_bc + log_keq_cd,
        ),
        edge_log_ratios=(log_keq_bc, log_keq_cd),
    )


def b43_concentrations(p_total, l_total, kd_ab, keq_bc, keq_cd):
    eq = _eq_43(p_total, l_total, kd_ab, keq_bc, keq_cd)
    pl1, pl2, pl3 = eq.complexes
    return {
        "p": eq.protein_free,
        "l": eq.ligand_free,
        "pl1": pl1,
        "pl2": pl2,
        "pl3": pl3,
    }


def b43_rates(p_total, l_total, kd_ab, keq_bc, keq_cd, koff_ab, kex_bc, kex_cd):
    eq = _eq_43(p_total, l_total, kd_ab, keq_bc, keq_cd)
    kbc, kcb = split_exchange_rate(kex_bc, 0.0, eq.edge_log_ratios[0])
    kcd, kdc = split_exchange_rate(kex_cd, 0.0, eq.edge_log_ratios[1])
    return {
        "kab": detailed_balance_rate(
            koff_ab, eq.log_populations[0], eq.log_populations[1]
        ),
        "kbc": kbc,
        "kcb": kcb,
        "kcd": kcd,
        "kdc": kdc,
    }


def b43_populations(p_total, l_total, kd_ab, keq_bc, keq_cd):
    pa, pb, pc, pd = _eq_43(p_total, l_total, kd_ab, keq_bc, keq_cd).populations
    return {"pa": pa, "pb": pb, "pc": pc, "pd": pd}


# --- settings_3st_binding_cs (conformational selection) ---------------------
def _eq_cs(p_total, l_total, kd_app, keq_ab):
    log_keq_ab = jnp.log(keq_ab)
    log_apo_weight = jnp.log1p(keq_ab)
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_protein_log_weights=(0.0, log_keq_ab),
        free_ligand_log_weights=(0.0,),
        complex_log_weights=(log_binding_weight(kd_app) + log_apo_weight,),
        edge_log_ratios=(log_keq_ab,),
    )


def cs_concentrations(p_total, l_total, kd_app, keq_ab):
    eq = _eq_cs(p_total, l_total, kd_app, keq_ab)
    a, b = eq.free_proteins
    return {"a": a, "b": b, "c": eq.complexes[0], "l": eq.ligand_free}


def cs_populations(p_total, l_total, kd_app, keq_ab):
    pa, pb, pc = _eq_cs(p_total, l_total, kd_app, keq_ab).populations
    return {"pa": pa, "pb": pb, "pc": pc}


def cs_conformational_rates(kex_ab, keq_ab):
    kab, kba = split_exchange_rate(kex_ab, 0.0, jnp.log(keq_ab))
    return {"kab": kab, "kba": kba}


def cs_binding_rates(p_total, l_total, kd_app, keq_ab, koff_bc):
    eq = _eq_cs(p_total, l_total, kd_app, keq_ab)
    return {
        "kbc": detailed_balance_rate(
            koff_bc, eq.log_populations[1], eq.log_populations[2]
        ),
        "kcb": koff_bc,
    }


def cs_intrinsic_values(kd_app, keq_ab):
    log_value = jnp.log(kd_app) + jnp.log(keq_ab) - jnp.log1p(keq_ab)
    return {"kd": report_only_positive_log_value(log_value)}


def cs_kon_values(koff_bc, kd_app, keq_ab):
    zero = koff_bc == 0.0
    log_kd_bc = jnp.log(kd_app) + jnp.log(keq_ab) - jnp.log1p(keq_ab)
    kon = report_only_positive_log_value(
        jnp.log(jnp.where(zero, 1.0, koff_bc)) - log_kd_bc
    )
    return {"kon": jnp.where(zero, 0.0, kon)}


# --- settings_3st_binding_if (induced fit) ----------------------------------
def _eq_if(p_total, l_total, kd_app, keq_bc):
    log_keq_bc = log_equilibrium_ratio(keq_bc)
    log_bound_partition = jnp.log1p(keq_bc)
    log_first_complex = log_binding_weight(kd_app) - log_bound_partition
    return solve_binding_equilibrium(
        p_total,
        l_total,
        free_ligand_log_weights=(0.0,),
        complex_log_weights=(
            log_first_complex,
            jnp.where(log_keq_bc == -jnp.inf, -jnp.inf, log_first_complex + log_keq_bc),
        ),
        edge_log_ratios=(log_keq_bc,),
    )


def if_concentrations(p_total, l_total, kd_app, keq_bc):
    eq = _eq_if(p_total, l_total, kd_app, keq_bc)
    return {
        "a": eq.free_proteins[0],
        "b": eq.complexes[0],
        "c": eq.complexes[1],
        "l": eq.ligand_free,
    }


def if_populations(p_total, l_total, kd_app, keq_bc):
    pa, pb, pc = _eq_if(p_total, l_total, kd_app, keq_bc).populations
    return {"pa": pa, "pb": pb, "pc": pc}


def if_conformational_rates(kex_bc, keq_bc):
    kbc, kcb = split_exchange_rate(kex_bc, 0.0, log_equilibrium_ratio(keq_bc))
    return {"kbc": kbc, "kcb": kcb}


def if_binding_rates(p_total, l_total, kd_app, keq_bc, koff_ab):
    eq = _eq_if(p_total, l_total, kd_app, keq_bc)
    return {
        "kab": detailed_balance_rate(
            koff_ab, eq.log_populations[0], eq.log_populations[1]
        ),
        "kba": koff_ab,
    }


def if_intrinsic_values(kd_app, keq_bc):
    return {"kd": report_only_positive_log_value(jnp.log(kd_app) + jnp.log1p(keq_bc))}


def if_kon_values(koff_ab, kd_app, keq_bc):
    zero = koff_ab == 0.0
    log_kd_ab = jnp.log(kd_app) + jnp.log1p(keq_bc)
    kon = report_only_positive_log_value(
        jnp.log(jnp.where(zero, 1.0, koff_ab)) - log_kd_ab
    )
    return {"kon": jnp.where(zero, 0.0, kon)}


def _key(function: Callable[..., Any]) -> Callable[..., Any]:
    return inspect.unwrap(function)


TWINS: dict[Callable[..., Any], Callable[..., Any]] = {
    _key(_b2.calculate_concentrations): b2_concentrations,
    _key(_b2.calculate_populations): b2_populations,
    _key(_b2.calculate_rates): b2_rates,
    _key(_b3p.calculate_concentrations): b3p_concentrations,
    _key(_b3p.calculate_populations): b3p_populations,
    _key(_b3p.calculate_rates): b3p_rates,
    _key(_b3d.calculate_concentrations): b3d_concentrations,
    _key(_b3d.calculate_populations): b3d_populations,
    _key(_b3d.calculate_rates): b3d_rates,
    _key(_b4p.calculate_concentrations): b4p_concentrations,
    _key(_b4p.calculate_populations): b4p_populations,
    _key(_b4p.calculate_rates): b4p_rates,
    _key(_b43.calculate_concentrations): b43_concentrations,
    _key(_b43.calculate_populations): b43_populations,
    _key(_b43.calculate_rates): b43_rates,
    _key(_bcs.calculate_concentrations): cs_concentrations,
    _key(_bcs.calculate_populations): cs_populations,
    _key(_bcs.calculate_conformational_rates): cs_conformational_rates,
    _key(_bcs.calculate_binding_rates): cs_binding_rates,
    _key(_bcs.calculate_intrinsic_values): cs_intrinsic_values,
    _key(_bcs.calculate_kon_values): cs_kon_values,
    _key(_bif.calculate_concentrations): if_concentrations,
    _key(_bif.calculate_populations): if_populations,
    _key(_bif.calculate_conformational_rates): if_conformational_rates,
    _key(_bif.calculate_binding_rates): if_binding_rates,
    _key(_bif.calculate_intrinsic_values): if_intrinsic_values,
    _key(_bif.calculate_kon_values): if_kon_values,
}
