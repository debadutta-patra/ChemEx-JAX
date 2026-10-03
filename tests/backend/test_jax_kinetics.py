# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Kinetic-model coverage: constraint program and profiles through JAX.

Every registered kinetic model, plain and with ``.mf``, ``.rs`` and ``.tc``,
on a synthetic CPMG 15N template (``tests/backend/synthetic/MODELS``) whose
conditions carry temperature, P_total != L_total and D2O so that every model
family builds:

* resolved values: ``evaluate_program`` (JAX, ``jit``) vs ChemEx's
  ``resolve`` ≤ 1e-10 relative per value (exact zeros must stay zero);
* program Jacobian: ``jacfwd`` vs Richardson central differences of
  ``resolve`` ≤ 1e-5 per derived value (effect-scaled; steps that leave the
  model's value domain are skipped, as ChemEx rejects them);
* profile from independent values: ≤ 1e-9 relative, ``jacfwd`` vs ``jacrev``
  ≤ 1e-10.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemex.parameters.parameterization import ParameterizationError
from tests.backend._checks import FD_STEPS, FORWARD_RTOL, JACREV_RTOL, relative_error
from tests.backend._examples import (
    QUICK_MODELS,
    build_model_case,
    model_cases,
    quick_or_full,
)
from tests.backend._jax_profile import make_program_function

pytestmark = pytest.mark.jax

MODELS = quick_or_full(model_cases(), QUICK_MODELS)
VALUE_RTOL = 1e-10
PROGRAM_GRADIENT_RTOL = 1e-5


def _program(example):
    import jax

    from chemex.backend import get_backend
    from chemex.parameters.program_evaluation import evaluate_program

    parameterization = example.parameterization
    ids = parameterization.independent_ids
    frame = dict(example.frame._items)
    x0 = np.array([frame[i] for i in ids])
    derived = parameterization.derived_ids

    def values(x):
        resolved = evaluate_program(
            parameterization, dict(zip(ids, x, strict=True)), get_backend("jax")
        )
        return jax.numpy.stack([resolved[d] for d in derived])

    return ids, derived, x0, values


@pytest.mark.parametrize("model", MODELS)
def test_resolved_values(model: str) -> None:
    import jax

    example = build_model_case(model)
    ids, derived, x0, values = _program(example)
    got = np.asarray(jax.jit(values)(x0))
    reference = np.array([example.values[d] for d in derived])
    bad = [
        (d, g, r)
        for d, g, r in zip(derived, got, reference, strict=True)
        if abs(g - r) > VALUE_RTOL * abs(r)
    ]
    assert not bad, (model, bad[:5])


@pytest.mark.parametrize("model", MODELS)
def test_program_jacobian(model: str) -> None:
    import jax

    example = build_model_case(model)
    parameterization = example.parameterization
    ids, derived, x0, values = _program(example)
    jacobian = np.asarray(jax.jit(jax.jacfwd(values))(x0))
    scales = np.where(x0 != 0.0, np.abs(x0), 1.0)
    effect = np.abs(jacobian) * scales  # derived x independent

    def resolved(x: np.ndarray) -> np.ndarray | None:
        try:
            frame = example.frame.with_updates(
                dict(zip(ids, map(float, x), strict=True))
            )
            out = parameterization.resolve(frame)
        except (ParameterizationError, ValueError, ArithmeticError):
            return None
        return np.array([out[d] for d in derived])

    for k in range(len(ids)):
        if not np.any(effect[:, k]):
            continue
        best = np.full(len(derived), np.inf)
        for step in FD_STEPS:
            h = step * scales[k]
            points = []
            for delta in (h, -h, h / 2, -h / 2):
                x = x0.copy()
                x[k] += delta
                points.append(resolved(x))
            if any(p is None for p in points):
                continue
            d_h = (points[0] - points[1]) / (2 * h)
            d_h2 = (points[2] - points[3]) / h
            fd = (4 * d_h2 - d_h) / 3
            best = np.minimum(best, np.abs(jacobian[:, k] - fd) * scales[k])
        rows = effect.max(axis=1)
        checked = np.isfinite(best) & (rows > 0)
        error = np.where(checked, best / np.where(rows > 0, rows, 1.0), 0.0)
        assert np.all(error <= PROGRAM_GRADIENT_RTOL), (
            model,
            ids[k],
            [(derived[i], error[i]) for i in np.argsort(error)[-3:]],
        )


@pytest.mark.parametrize("model", MODELS)
def test_profile_from_independent_values(model: str) -> None:
    import jax

    example = build_model_case(model)
    profile = example.profiles[0]
    ids, f = make_program_function(profile, example.parameterization, example.frame)
    frame = dict(example.frame._items)
    x0 = np.array([frame[i] for i in ids])
    reference = np.asarray(profile.calculate_unscaled(example.values), float)
    assert relative_error(np.asarray(jax.jit(f)(x0)), reference) <= FORWARD_RTOL
    jac_fwd = np.asarray(jax.jit(jax.jacfwd(f))(x0))
    jac_rev = np.asarray(jax.jit(jax.jacrev(f))(x0))
    scale = float(np.max(np.abs(jac_fwd))) or 1.0
    assert np.max(np.abs(jac_fwd - jac_rev)) / scale <= JACREV_RTOL
