# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Public ``chemex.jax`` API: residual parity with native fitting, and safety.

Residual parity criterion (every case): χ² relative error ≤ 1e-9, and
``max|Δr| / max|s·c/e|`` ≤ 1e-9, i.e. residual errors measured on the scale of
the weighted calculated signal.  ``max|Δr| / max|r|`` is reported, not
asserted: for a well-fitting high-SNR profile it amplifies the (≤ 1e-9)
forward error by ``max|s·c/e| / max|r|`` (≈ 3e1 for CEST_15N_CW, whose
dephased, decoupled Liouvillian limits both backends to ~4e-11).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from tests.backend._examples import (
    QUICK_CASES,
    all_cases,
    build_case,
    build_model_case,
    quick_or_full,
)

pytestmark = pytest.mark.jax

RESIDUAL_RTOL = 1e-9


def _native(example):
    from chemex.evaluation.native import (
        EvaluationEngine,
        EvaluationFrame,
        EvaluationResult,
    )

    parameterization = example.parameterization
    engine = EvaluationEngine.from_experiments(example.experiments, parameterization)
    result = engine.new_evaluator().evaluate(
        EvaluationFrame.from_lifecycle_frame(parameterization, example.frame)
    )
    assert isinstance(result, EvaluationResult), result
    return result


def _weighted_signal(example, native) -> float:
    profiles = [p for e in example.experiments for p in e.profiles]
    normalized = np.asarray(native.normalized_calculations)
    offset, largest = 0, 0.0
    for profile in profiles:
        n = profile.data.exp.size
        mask = np.asarray(profile.data.mask)
        weighted = normalized[offset : offset + n][mask] / profile.data.err[mask]
        largest = max(largest, float(np.max(np.abs(weighted), initial=0.0)))
        offset += n
    return largest


@pytest.mark.parametrize("case", quick_or_full(all_cases(), QUICK_CASES))
def test_residuals_match_native_fitting(case: str) -> None:
    import jax

    import chemex.jax as cj

    example = build_case(case)
    native = _native(example)
    residuals = cj.compile_residuals(
        example.experiments,
        example.parameterization,
        example.parameterization.independent_ids,
        frame=example.frame,
    )
    got = np.asarray(jax.jit(residuals)(residuals.x0))
    expected = np.asarray(native.residuals)
    assert got.shape == expected.shape
    chi2, chi2_native = float(got @ got), float(expected @ expected)
    assert abs(chi2 - chi2_native) <= RESIDUAL_RTOL * chi2_native
    scale = _weighted_signal(example, native)
    assert np.max(np.abs(got - expected)) <= RESIDUAL_RTOL * scale


@pytest.mark.parametrize("case", ["Experiments/CPMG_15N_IP", "Experiments/CEST_15N"])
def test_compile_profile_matches_calculate_unscaled(case: str) -> None:
    import jax

    import chemex.jax as cj

    example = build_case(case)
    parameterization = example.parameterization
    for profile in example.sample_profiles(2):
        from chemex.parameters.program_evaluation import independent_dependencies

        free = independent_dependencies(parameterization, profile.name_map.values())
        compiled = cj.compile_profile(
            profile, parameterization, free, frame=example.frame
        )
        reference = np.asarray(profile.calculate_unscaled(example.values))
        got = np.asarray(jax.jit(compiled)(compiled.x0))
        assert np.max(np.abs(got - reference)) <= 1e-9 * np.max(np.abs(reference))
        jac_fwd = np.asarray(jax.jacfwd(compiled)(compiled.x0))
        jac_rev = np.asarray(jax.jacrev(compiled)(compiled.x0))
        assert np.max(np.abs(jac_fwd - jac_rev)) <= 1e-10 * np.max(np.abs(jac_fwd))
        # A subset of free parameters: the rest stay at ``frame``.
        subset = cj.compile_profile(
            profile, parameterization, free[:2], frame=example.frame
        )
        np.testing.assert_allclose(np.asarray(subset(subset.x0)), got, rtol=0, atol=0)


def test_profiles_in_a_group_share_one_kernel_and_match_individually() -> None:
    import jax

    import chemex.jax as cj

    example = build_case("Experiments/CPMG_15N_IP")
    parameterization = example.parameterization
    residuals = cj.compile_residuals(
        example.experiments,
        parameterization,
        parameterization.independent_ids,
        frame=example.frame,
    )
    assert residuals.group_count == 2  # 108 profiles at two fields
    profiles = [p for e in example.experiments for p in e.profiles]
    signatures = {cj.kernel_signature(p) for p in profiles}
    assert len(signatures) == 2
    first, *others = [
        p
        for p in profiles
        if cj.kernel_signature(p) == cj.kernel_signature(profiles[0])
    ][:4]
    compiled = [
        cj.compile_profile(
            p, parameterization, parameterization.independent_ids, frame=example.frame
        )
        for p in (first, *others)
    ]
    assert len({id(c._kernel) for c in compiled}) == 1  # one cached kernel
    for c, p in zip(compiled, (first, *others), strict=True):
        got = np.asarray(jax.jit(c)(c.x0))
        reference = np.asarray(p.calculate_unscaled(example.values))
        assert np.max(np.abs(got - reference)) <= 1e-9 * np.max(np.abs(reference))
    again = cj.compile_residuals(
        example.experiments,
        parameterization,
        parameterization.independent_ids,
        frame=example.frame,
    )
    assert [g.kernel for g in again._groups] == [g.kernel for g in residuals._groups]


def test_inputs_must_be_float64_vectors() -> None:
    import jax.numpy as jnp

    import chemex.jax as cj

    example = build_case("Experiments/RELAXATION_NZ")
    residuals = cj.compile_residuals(
        example.experiments,
        example.parameterization,
        example.parameterization.independent_ids,
        frame=example.frame,
    )
    with pytest.raises(TypeError, match="float64"):
        residuals(jnp.asarray(residuals.x0, dtype=jnp.float32))
    with pytest.raises(ValueError, match="free parameter values"):
        residuals(residuals.x0[:-1])
    with pytest.raises(ValueError, match="independent"):
        cj.compile_residuals(
            example.experiments,
            example.parameterization,
            example.parameterization.derived_ids[:1],
            frame=example.frame,
        )


def test_validate_runs_chemex_domain_checks() -> None:
    import chemex.jax as cj
    from chemex.parameters.parameterization import ParameterizationError

    example = build_model_case("3st")
    parameterization = example.parameterization
    free = tuple(
        i for i in parameterization.independent_ids if i.startswith(("__PB", "__PC"))
    )
    residuals = cj.compile_residuals(
        example.experiments, parameterization, free, frame=example.frame
    )
    residuals.validate(residuals.x0)
    with pytest.raises(ParameterizationError):
        residuals.validate([0.7, 0.6])


@pytest.mark.memory_heavy
def test_jax_then_numpy_and_native_evaluation_unchanged() -> None:
    import jax

    import chemex.jax as cj

    example = build_case("Experiments/CEST_15N")
    before = np.array(_native(example).residuals, copy=True)
    profile = example.profiles[0]
    unscaled_before = np.array(profile.calculate_unscaled(example.values), copy=True)
    residuals = cj.compile_residuals(
        example.experiments,
        example.parameterization,
        example.parameterization.independent_ids,
        frame=example.frame,
    )
    x0 = residuals.x0
    jax.jit(jax.grad(residuals.chi2))(x0).block_until_ready()
    jax.jit(jax.jacrev(residuals))(x0).block_until_ready()
    jax.jit(jax.vmap(residuals))(np.stack([x0, x0])).block_until_ready()
    assert all(
        type(v) is not jax.Array
        for p in example.profiles
        for v in p.spectrometer.par_values.values()
    )
    np.testing.assert_array_equal(np.asarray(_native(example).residuals), before)
    np.testing.assert_array_equal(
        np.asarray(profile.calculate_unscaled(example.values)), unscaled_before
    )


@pytest.mark.memory_heavy
def test_concurrent_use_from_threads_matches_serial() -> None:
    import jax

    import chemex.jax as cj

    cases = [
        "Experiments/RELAXATION_NZ",
        "Experiments/CPMG_13C_IP",
        "Experiments/CEST_15N",
    ]
    examples = {case: build_case(case) for case in cases}

    def evaluate(case: str) -> np.ndarray:
        example = examples[case]
        residuals = cj.compile_residuals(
            example.experiments,
            example.parameterization,
            example.parameterization.independent_ids,
            frame=example.frame,
        )
        return np.asarray(jax.jit(jax.grad(residuals.chi2))(residuals.x0))

    serial = {case: evaluate(case) for case in cases}
    results: dict[tuple[int, str], np.ndarray] = {}
    errors: list[BaseException] = []

    def worker(index: int, case: str) -> None:
        try:
            results[index, case] = evaluate(case)
        except BaseException as error:  # noqa: BLE001 - surface in the main thread
            errors.append(error)

    threads = [
        threading.Thread(target=worker, args=(i, case))
        for i in range(2)
        for case in cases
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for (_, case), value in results.items():
        np.testing.assert_array_equal(value, serial[case])


@pytest.mark.parametrize(
    ("mode", "chunk"), [("forward", 3), ("forward", 7), ("reverse", 64)]
)
def test_chunked_jacobian_matches_unchunked(mode: str, chunk: int) -> None:
    import jax

    import chemex.jax as cj

    example = build_case("Experiments/CPMG_13C_IP")
    residuals = cj.compile_residuals(
        example.experiments,
        example.parameterization,
        example.parameterization.independent_ids[:10],
        frame=example.frame,
    )
    x0 = residuals.x0
    unchunked = (jax.jacfwd if mode == "forward" else jax.jacrev)(residuals)
    full = np.asarray(unchunked(x0))
    chunked = np.asarray(cj.jacobian(residuals, x0, chunk_size=chunk, mode=mode))
    assert chunked.shape == full.shape
    # Chunking changes nothing but grouping; forward vs reverse mode differ at
    # the suite's jacfwd/jacrev tolerance.
    assert np.max(np.abs(chunked - full)) <= 1e-13 * np.max(np.abs(full))
    forward = np.asarray(jax.jacfwd(residuals)(x0))
    assert np.max(np.abs(chunked - forward)) <= 1e-10 * np.max(np.abs(forward))
    np.testing.assert_array_equal(
        np.asarray(cj.jacobian(residuals, x0, mode=mode)), full
    )
    with pytest.raises(ValueError, match="mode"):
        cj.jacobian(residuals, x0, mode="sideways")
    with pytest.raises(ValueError, match="chunk_size"):
        cj.jacobian(residuals, x0, chunk_size=0)
