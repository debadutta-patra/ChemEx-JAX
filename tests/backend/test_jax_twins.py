# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Scientific-function twins vs their NumPy originals over argument grids.

Values: every output component ≤ 1e-10 relative (exact zeros stay zero) at
every grid point where ChemEx accepts the arguments.

Gradients: ``jacfwd`` vs Richardson central differences of the NumPy function,
≤ 1e-5 of the output's effect scale *plus the FD reference's own roundoff
budget* (``FD_NOISE * eps * max|f| / h``): at extreme grid corners (KD 1e-9
with micromolar ligand) a component can be 1e-3 while its derivative is tiny,
and float64 FD cannot certify anything below its noise floor there.  Those
corners are covered by exact oracles instead: closed-form 2st binding and
mass balances for the oligomer models solved in mpmath (90 digits), plus
ChemEx's own analytic Eyring partials.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.jax

VALUE_RTOL = 1e-10
GRADIENT_RTOL = 1e-5
FD_STEPS = (1e-3, 1e-4, 1e-5)
FD_NOISE = 64.0
EPS = float(np.finfo(np.float64).eps)
EXACT_RTOL = 1e-10
SAMPLES = 48

# Argument values by parameter name: structural zeros, equal pools, weak and
# strong binding, slow and fast exchange.
ARGUMENT_VALUES: dict[str, tuple[float, ...]] = {
    "p_total": (1e-6, 1.0e-3, 2.0e-3),
    "l_total": (0.0, 1e-6, 1.0e-3, 5e-3),
    "kd": (1e-9, 1e-6, 1e-4, 1e-2, 1.0),
    "keq": (0.0, 0.05, 1.0, 20.0),
    "koff": (0.0, 5.0, 500.0),
    "kex": (0.0, 30.0, 3000.0),
}


def _values_for(name: str) -> tuple[float, ...]:
    for prefix, values in ARGUMENT_VALUES.items():
        if name == prefix or name.startswith(prefix):
            return values
    msg = f"no grid for argument {name!r}"
    raise KeyError(msg)


def _grid(function: Callable[..., Any]) -> list[tuple[float, ...]]:
    names = list(inspect.signature(inspect.unwrap(function)).parameters)
    axes = [_values_for(n) for n in names]
    points = list(itertools.product(*axes))
    rng = np.random.default_rng(20261002)
    if len(points) > SAMPLES:
        points = [points[i] for i in rng.choice(len(points), SAMPLES, replace=False)]
    return points


def _numpy(function: Callable[..., Any], args: tuple[float, ...]) -> dict | None:
    try:
        out = inspect.unwrap(function)(*args)
    except (ValueError, ArithmeticError, RuntimeError):
        return None
    return dict(out)


_GRID_MODELS = (
    "2st_binding",
    "3st_binding_partner_2st",
    "3st_double_binding",
    "4st_binding_partner_2st",
    "4st_binding_3_bound_states",
    "3st_binding_cs",
    "3st_binding_if",
    "2st_monomer_dimer",
    "2st_monomer_trimer",
    "2st_monomer_tetramer",
    "3st_monomer_dimer_trimer",
    "3st_monomer_dimer_tetramer",
)


def _binding_and_oligomer_functions() -> list[Callable[..., Any]]:
    """Registered binding/oligomer user functions, collected without JAX."""
    from chemex.parameters.userfunctions import user_function_registry
    from chemex.runtime import ensure_plugins_registered

    ensure_plugins_registered()
    functions: dict[Callable[..., Any], None] = {}
    for model in _GRID_MODELS:
        for function_id, function in user_function_registry.get(model).items():
            if not function_id.startswith("pop_"):
                functions[inspect.unwrap(function)] = None
    return list(functions)


def _id(function: Callable[..., Any]) -> str:
    return f"{function.__module__.rsplit('.', 1)[-1]}.{function.__name__}"


@pytest.mark.parametrize("function", _binding_and_oligomer_functions(), ids=_id)
def test_twin_values_on_grid(function: Callable[..., Any]) -> None:
    from chemex.backend.jax_scientific import twin_for

    twin = twin_for(function)
    assert twin is not None
    checked = 0
    for args in _grid(function):
        expected = _numpy(function, args)
        if expected is None:
            continue
        got = twin(*args)
        assert set(got) == set(expected), (args, set(got), set(expected))
        for key, value in expected.items():
            actual = float(np.asarray(got[key]))
            if np.isnan(value):
                assert np.isnan(actual), (args, key)
                continue
            assert abs(actual - value) <= VALUE_RTOL * abs(value), (
                _id(function),
                args,
                key,
                actual,
                value,
            )
        checked += 1
    assert checked > 0, "no admissible grid point"


@pytest.mark.parametrize("function", _binding_and_oligomer_functions(), ids=_id)
def test_twin_gradients_on_grid(function: Callable[..., Any]) -> None:
    import jax
    import jax.numpy as jnp

    from chemex.backend.jax_scientific import twin_for

    twin = twin_for(function)
    names = list(inspect.signature(inspect.unwrap(function)).parameters)
    # Differentiate with respect to model parameters, not the condition totals.
    free = [i for i, n in enumerate(names) if n not in ("p_total", "l_total")]
    points = [
        a
        for a in _grid(function)
        if all(a[i] > 0.0 for i in free) and _numpy(function, a) is not None
    ][:12]
    for args in points:
        expected = _numpy(function, args)
        keys = sorted(expected)

        def vector(
            x: Any, args: tuple[float, ...] = args, keys: list[str] = keys
        ) -> Any:
            full = list(args)
            for position, i in enumerate(free):
                full[i] = x[position]
            out = twin(*full)
            return jnp.stack([jnp.asarray(out[k], dtype=jnp.float64) for k in keys])

        x0 = np.array([args[i] for i in free])
        jacobian = np.asarray(jax.jacfwd(vector)(x0))
        for k, i in enumerate(free):
            best = np.full(len(keys), np.inf)
            noise = np.zeros(len(keys))
            for step in FD_STEPS:
                h = step * x0[k]
                rows = []
                for delta in (h, -h, h / 2, -h / 2):
                    shifted = list(args)
                    shifted[i] += delta
                    out = _numpy(function, tuple(shifted))
                    rows.append(
                        None if out is None else np.array([out[q] for q in keys])
                    )
                if any(r is None for r in rows):
                    continue
                fd = (4 * (rows[2] - rows[3]) / h - (rows[0] - rows[1]) / (2 * h)) / 3
                error = np.abs(jacobian[:, k] - fd) * x0[k]
                floor = FD_NOISE * EPS * np.max(np.abs(rows), axis=0) / h * x0[k]
                better = error - floor < best - noise
                best = np.where(better, error, best)
                noise = np.where(better, floor, noise)
            effect = np.abs(jacobian) * x0
            scale = effect.max(axis=1)
            ok = (
                ~np.isfinite(best)
                | (scale == 0)
                | (best <= GRADIENT_RTOL * scale + noise)
            )
            assert np.all(ok), (
                _id(function),
                args,
                names[i],
                best / np.where(scale > 0, scale, 1),
            )


def test_pair_rate_and_population_edge_cases() -> None:
    from chemex.backend import jax_scientific as js
    from chemex.models import constraints
    from chemex.models.kinetic import settings_nst

    cases = [
        (0.0, 0.3, 0.2),
        (100.0, 0.0, 0.4),
        (100.0, 0.4, 0.0),
        (100.0, 0.25, 0.25),
        (100.0, 0.1, 0.3),
        (100.0, 1e-300, 0.5),
    ]
    for kex, p_i, p_j in cases:
        expected = settings_nst.calculate_pair_rates(kex, p_i, p_j)
        got = js.pair_rates(kex, p_i, p_j)
        for key in expected:
            assert float(got[key]) == pytest.approx(
                expected[key], rel=VALUE_RTOL, abs=0.0
            )
    for rates in [(0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (3.0, 7.0)]:
        expected = constraints.pop_2st.__wrapped__(*rates)
        got = js.pop_2st(*rates)
        for key in expected:
            assert float(got[key]) == pytest.approx(
                expected[key], rel=VALUE_RTOL, abs=0.0
            )
    for rates in [
        (0, 0, 0, 0, 2.0, 3.0),
        (0, 0, 4.0, 1.0, 0, 0),
        (2.0, 5.0, 0, 0, 0, 0),
        (1.0, 9.0, 2.0, 8.0, 3.0, 7.0),
        (1.0, 9.0, 0, 0, 3.0, 7.0),
    ]:
        expected = constraints.pop_3st.__wrapped__(*map(float, rates))
        got = js.pop_3st(*map(float, rates))
        for key in expected:
            assert float(got[key]) == pytest.approx(
                expected[key], rel=VALUE_RTOL, abs=0.0
            )
    for rates in [
        (0,) * 6 + (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        tuple(float(i + 1) for i in range(12)),
    ]:
        expected = constraints.pop_4st.__wrapped__(*map(float, rates))
        got = js.pop_4st(*map(float, rates))
        for key in expected:
            assert float(got[key]) == pytest.approx(
                expected[key], rel=VALUE_RTOL, abs=1e-15
            )


def test_eyring_twins_match_analytic_partials() -> None:
    import jax

    from chemex.backend import jax_scientific as js
    from chemex.models.kinetic import _eyring

    args = (2.0e3, -5.0, 6.5e4, 12.0, 25.0)
    rate = float(js.eyring_rate(*args)["rate"])
    assert rate == pytest.approx(
        _eyring.calculate_rate_component(*args)["rate"], rel=VALUE_RTOL
    )
    grad = jax.grad(lambda *a: js.eyring_rate(*a)["rate"], argnums=tuple(range(5)))(
        *args
    )
    for got, partial in zip(grad, _eyring.EYRING_RATE_PARTIALS, strict=True):
        assert float(got) == pytest.approx(partial(*args), rel=1e-10)

    states = ("pa", "pb", "pc")
    partials = _eyring.thermodynamic_population_partials(states)
    pop_args = (8.0e3, 20.0, -3.0e3, -12.0, 15.0)
    reference = js.populations_3st_eyring(*pop_args)
    for name in states:
        g = jax.grad(
            lambda *a, n=name: js.populations_3st_eyring(*a)[n], argnums=tuple(range(5))
        )(*pop_args)
        for got, partial in zip(g, partials[name], strict=True):
            assert float(got) == pytest.approx(partial(*pop_args), rel=1e-9, abs=1e-18)
        assert float(reference[name]) > 0.0


def _mp_derivative(function: Callable[[Any], Any], x: float) -> list[float]:
    import mpmath as mp

    # 90 digits and a 1e-30 relative step: components such as pa = 1 - 4e-20
    # lose ~45 digits to cancellation in the difference quotient.
    h = mp.mpf(x) * mp.mpf("1e-30")
    plus, minus = function(mp.mpf(x) + h), function(mp.mpf(x) - h)
    return [float((a - b) / (2 * h)) for a, b in zip(plus, minus, strict=True)]


@pytest.mark.parametrize(
    "point",
    [
        (1e-3, 1e-6, 1e-9),
        (1e-6, 1e-3, 1e-9),
        (1e-3, 1e-3, 1e-9),
        (1e-3, 5e-3, 1e-4),
        (2e-3, 1e-3, 1.0),
        (1e-3, 1e-6, 1e-4),
    ],
)
def test_binding_derivatives_exact(point: tuple[float, float, float]) -> None:
    """2st binding vs the closed-form quadratic root at 90 digits."""
    import jax
    import jax.numpy as jnp
    import mpmath as mp

    from chemex.backend import jax_binding as jb

    p_total, l_total, kd = point
    mp.mp.dps = 90

    def exact(k: Any) -> list[Any]:
        p, lig = mp.mpf(p_total), mp.mpf(l_total)
        s = p + lig + k
        bound = (s - mp.sqrt(s * s - 4 * p * lig)) / 2
        return [p - bound, lig - bound, bound]

    keys = ("p_free", "l_free", "pl")
    got = jax.jacfwd(
        lambda k: jnp.stack(
            [jb.b2_concentrations(p_total, l_total, k)[n] for n in keys]
        )
    )(kd)
    for actual, expected in zip(
        np.asarray(got), _mp_derivative(exact, kd), strict=True
    ):
        assert actual == pytest.approx(expected, rel=EXACT_RTOL)


@pytest.mark.parametrize("p_total", [1e-6, 1e-3, 1.0])
@pytest.mark.parametrize("kd", [1e-9, 1e-5, 1e-3, 1e2])
@pytest.mark.parametrize(
    ("module_name", "stoichiometries"),
    [
        ("settings_2st_monomer_dimer", ((2, 1, 1),)),
        ("settings_2st_monomer_trimer", ((3, 2, 1),)),
        ("settings_2st_monomer_tetramer", ((4, 3, 1),)),
        ("settings_3st_monomer_dimer_trimer", ((2, 1, 1), (3, 2, 2))),
    ],
)
def test_oligomer_derivatives_exact(
    module_name: str, stoichiometries: tuple, p_total: float, kd: float
) -> None:
    """Oligomer populations vs the mass balance solved in mpmath (90 digits).

    Derivatives are taken with respect to KD (both KDs set to ``kd`` for the
    3-state model), exercising the implicit-function ``custom_jvp``.
    """
    import importlib

    import jax
    import jax.numpy as jnp
    import mpmath as mp

    from chemex.backend.jax_scientific import twin_for

    mp.mp.dps = 90
    module = importlib.import_module(f"chemex.models.kinetic.{module_name}")
    twin = twin_for(module.calculate_populations)
    n_kd = 1 if len(stoichiometries) == 1 else 2
    keys = ("pa", "pb", "pc")[: len(stoichiometries) + 1]

    def exact(k: Any) -> list[Any]:
        p = mp.mpf(p_total)
        # Tagged fraction of each oligomer: n * p**(n-1) * x**n / prod(KD); total 1.
        coefficients = [
            n * p ** (power) / k**kd_count for n, power, kd_count in stoichiometries
        ]

        powers = [n for n, _, _ in stoichiometries]
        # Newton on g(y) = log(e^y + sum c e^{n y}) from y = 0: g is convex and
        # increasing with g(0) >= 0, so the iterates decrease monotonically.
        y = mp.mpf(0)
        for _ in range(500):
            terms = [mp.exp(y)] + [
                c * mp.exp(n * y) for c, n in zip(coefficients, powers, strict=True)
            ]
            total = mp.fsum(terms)
            weighted = [terms[0]] + [
                n * t for n, t in zip(powers, terms[1:], strict=True)
            ]
            step = mp.log(total) / (mp.fsum(weighted) / total)
            y -= step
            if abs(step) < mp.mpf("1e-85"):
                break
        x = mp.exp(y)
        return [x] + [
            c * x**n for c, (n, _, _) in zip(coefficients, stoichiometries, strict=True)
        ]

    def populations(k: Any) -> Any:
        out = twin(p_total, *([k] * n_kd))
        return jnp.stack([out[q] for q in keys])

    values = np.asarray(populations(kd))
    expected_values = [float(v) for v in exact(mp.mpf(kd))]
    np.testing.assert_allclose(values, expected_values, rtol=1e-12, atol=1e-300)
    got = np.asarray(jax.jacfwd(populations)(kd))
    expected = _mp_derivative(exact, kd)
    scale = max(abs(e) for e in expected) or 1.0
    for actual, wanted in zip(got, expected, strict=True):
        assert abs(actual - wanted) <= EXACT_RTOL * scale, (actual, wanted)
