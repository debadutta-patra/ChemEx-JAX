# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Backend-generic constraint-program evaluation.

* NumPy backend: bit-identical to ChemEx's ``resolve`` on every case.
* Traced path: no domain checks inside; ``validate_concrete`` (ChemEx's own
  resolver) rejects inadmissible values, so no check is dropped.
* Functions without a twin fail loudly; JAX evaluation leaves NumPy caches and
  results untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemex.parameters.parameterization import ParameterizationError
from chemex.parameters.program_evaluation import (
    evaluate_program,
    independent_dependencies,
    required_constraints,
    validate_concrete,
)
from tests.backend._examples import all_cases, build_case, build_model_case


@pytest.mark.parametrize("case", all_cases())
def test_numpy_evaluation_is_bit_identical_to_resolve(case: str) -> None:
    example = build_case(case)
    got = evaluate_program(example.parameterization, dict(example.frame._items))
    for param_id, value in example.values.items():
        assert np.float64(got[param_id]).tobytes() == np.float64(value).tobytes(), (
            param_id
        )


def test_targets_restrict_to_required_constraints() -> None:
    example = build_case("Experiments/CPMG_15N_IP")
    parameterization = example.parameterization
    profile = example.profiles[0]
    targets = tuple(profile.name_map.values())
    needed = required_constraints(parameterization, targets)
    assert 0 < len(needed) < len(parameterization.ordered_constraints)
    ids = independent_dependencies(parameterization, targets)
    frame = dict(example.frame._items)
    got = evaluate_program(
        parameterization, {i: frame[i] for i in ids}, targets=targets
    )
    for target in targets:
        assert got[target] == example.values[target]


def test_validate_concrete_rejects_what_tracing_cannot_check() -> None:
    from chemex.backend import get_backend

    pytest.importorskip("jax")
    example = build_model_case("3st")
    parameterization = example.parameterization
    frame = dict(example.frame._items)
    populations = [
        i for i in parameterization.independent_ids if i.startswith(("__PB", "__PC"))
    ]
    assert len(populations) == 2
    invalid = {populations[0]: 0.7, populations[1]: 0.6}  # sum > 1: PA < 0
    values = evaluate_program(parameterization, frame | invalid, get_backend("jax"))
    assert all(np.isfinite(float(v)) for v in values.values())
    with pytest.raises(ParameterizationError):
        validate_concrete(parameterization, example.frame, invalid)


@pytest.mark.jax
def test_missing_twin_fails_loudly() -> None:
    from chemex.backend.jax_scientific import MissingTwinError, evaluate

    def unknown_function(x: float) -> dict[str, float]:
        return {"y": x}

    with pytest.raises(MissingTwinError, match="no JAX twin"):
        evaluate("unknown", unknown_function, (1.0,))


@pytest.mark.jax
@pytest.mark.parametrize(
    "model", ["2st_binding", "2st_eyring", "3st_monomer_dimer_trimer", "4st", "2st_hd"]
)
def test_jax_then_numpy_leaves_cached_functions_unchanged(model: str) -> None:
    import jax

    from chemex.backend import get_backend

    example = build_model_case(model)
    parameterization = example.parameterization
    ids = parameterization.independent_ids
    frame = dict(example.frame._items)
    before = dict(parameterization.resolve(example.frame))
    x0 = np.array([frame[i] for i in ids])

    def total(x: object) -> object:
        values = evaluate_program(
            parameterization, dict(zip(ids, list(x), strict=True)), get_backend("jax")
        )
        return sum(values[d] for d in parameterization.derived_ids)

    jax.jit(jax.grad(total))(x0).block_until_ready()
    jax.jit(jax.vmap(total))(np.stack([x0, x0])).block_until_ready()
    after = dict(parameterization.resolve(example.frame))
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert np.float64(after[key]).tobytes() == np.float64(value).tobytes(), key
