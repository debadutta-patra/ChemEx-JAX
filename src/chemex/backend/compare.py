# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""``chemex compare-backends``: check the JAX backend against NumPy on your data.

Builds the experiments exactly as ``chemex fit`` does (same ``-e``/``-p``/
``-d``/``--include``/``--exclude``), evaluates them with ChemEx's native NumPy
evaluator and with :mod:`chemex.jax`, and reports per experiment:

* agreement of the unscaled calculated intensities (max relative error over
  profiles: ``max|Δc| / max|c|``);
* agreement of the weighted residuals with the native fit residuals
  (``max|Δr| / max|s·c/e|``) and of χ²;
* NumPy calculation time vs JAX compile and run time;
* optionally (``--gradients N``) JAX derivatives vs finite differences of the
  native residuals for the N most widely shared fitted parameters.

Exits with status 1 when a parity check exceeds ``--rtol``.  JAX is imported
only when the command runs; ``import chemex`` never needs it.
"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser, Namespace, _SubParsersAction
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from chemex.parameters.spin_system import SpinSystem

COMMAND = "compare-backends"


class NativeEvaluationError(RuntimeError):
    """ChemEx's native evaluator rejected the parameter values."""


FD_STEPS = (1e-3, 1e-4, 1e-5)
# Below this effect (max|dr/dθ|·|θ| / max|r|) finite differences are noise.
FD_EFFECT_FLOOR = 1e-6


def add_parser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Register ``chemex compare-backends`` (called from ``chemex.cli``)."""
    parser = subparsers.add_parser(
        COMMAND,
        help="Compare the NumPy and JAX backends on your data (needs JAX)",
        description=__doc__.splitlines()[0],
    )
    parser.set_defaults(func=compare_backends, analysis_command=False)
    parser.add_argument(
        "-e",
        "--experiments",
        type=Path,
        metavar="FILE",
        nargs="+",
        required=True,
        help="Experiment file(s), as for 'chemex fit'",
    )
    parser.add_argument(
        "-p",
        "--parameters",
        type=Path,
        metavar="FILE",
        nargs="+",
        required=True,
        help="Parameter file(s), as for 'chemex fit' (later files override)",
    )
    parser.add_argument(
        "-d",
        "--model",
        metavar="MODEL",
        default="2st",
        help="Exchange model (default: '2st')",
    )
    parser.add_argument(
        "--include",
        metavar="ID",
        nargs="+",
        type=SpinSystem.from_name,
        help="Residue(s) to include",
    )
    parser.add_argument(
        "--exclude",
        metavar="ID",
        nargs="+",
        type=SpinSystem.from_name,
        help="Residue(s) to exclude",
    )
    parser.add_argument(
        "--gradients",
        type=int,
        default=0,
        metavar="N",
        help="Also compare derivatives for the N most widely shared fitted "
        "parameters (default: 0, off)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        metavar="K",
        help="Jacobian columns computed at once for --gradients (default: 16)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance of the parity checks (default: 1e-9)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="FILE",
        help="Also write the report as JSON",
    )
    return parser


def _build(args: Namespace):
    from chemex.configuration.methods import Method, Selection
    from chemex.configuration.parameters import read_defaults
    from chemex.experiments.builder import build_experiments
    from chemex.runtime import AnalysisSession

    session = AnalysisSession.create()
    session.set_model(args.model)
    experiments = build_experiments(
        args.experiments, Selection(args.include, args.exclude), session=session
    )
    session.parameters.set_defaults(read_defaults(args.parameters))
    if not session.try_build_analysis_values():
        msg = "Parameter initialisation failed; run 'chemex fit' for details"
        raise RuntimeError(msg)
    parameterization = session.compile_parameterization(Method(), experiments.param_ids)
    frame = parameterization.frame_from_snapshot(session.analysis_values.snapshot())
    return experiments, parameterization, frame


def _single(experiment: Any) -> Any:
    from chemex.containers.experiments import Experiments

    container = Experiments()
    container.add(experiment)
    return container


def _native(experiments: Any, parameterization: Any, frame: Any) -> Any:
    from chemex.evaluation.native import (
        EvaluationEngine,
        EvaluationFrame,
        EvaluationResult,
    )

    engine = EvaluationEngine.from_experiments(experiments, parameterization)
    evaluator = engine.new_evaluator()
    result = evaluator.evaluate(
        EvaluationFrame.from_lifecycle_frame(parameterization, frame)
    )
    if not isinstance(result, EvaluationResult):
        raise NativeEvaluationError(str(result))
    return result, evaluator


def _compare_experiment(experiment, parameterization, frame, free, rtol) -> dict:
    import jax

    import chemex.jax as cj

    container = _single(experiment)
    profiles = list(experiment.profiles)
    native, _ = _native(container, parameterization, frame)
    values = parameterization.resolve(frame)

    start = time.perf_counter()
    for profile in profiles:
        profile.calculate_unscaled(values)
    numpy_time = time.perf_counter() - start

    residuals = cj.compile_residuals(container, parameterization, free, frame=frame)
    calc_fn, res_fn = jax.jit(residuals.calculations), jax.jit(residuals)
    start = time.perf_counter()
    calc = np.asarray(calc_fn(residuals.x0))
    res = np.asarray(res_fn(residuals.x0))
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    jax.block_until_ready(calc_fn(residuals.x0))
    jax_time = time.perf_counter() - start

    native_calc = np.asarray(native.unscaled_calculations)
    native_res = np.asarray(native.residuals)
    normalized = np.asarray(native.normalized_calculations)
    forward, signal, offset = 0.0, 0.0, 0
    for profile in profiles:
        n = profile.data.exp.size
        ref = native_calc[offset : offset + n]
        scale = float(np.max(np.abs(ref))) or 1.0
        forward = max(
            forward, float(np.max(np.abs(calc[offset : offset + n] - ref))) / scale
        )
        mask = np.asarray(profile.data.mask)
        weighted = normalized[offset : offset + n][mask] / profile.data.err[mask]
        signal = max(signal, float(np.max(np.abs(weighted), initial=0.0)))
        offset += n
    chi2_native = float(native_res @ native_res)
    chi2_jax = float(res @ res)
    residual = float(np.max(np.abs(res - native_res), initial=0.0)) / (signal or 1.0)
    chi2_error = abs(chi2_jax - chi2_native) / (chi2_native or 1.0)
    cj.release_memory()
    return {
        "experiment": str(experiment.filename),
        "type": experiment.name,
        "profiles": len(profiles),
        "kernels": residuals.group_count,
        "forward_error": forward,
        "residual_error": residual,
        "chi2_native": chi2_native,
        "chi2_jax": chi2_jax,
        "chi2_error": chi2_error,
        "numpy_seconds": numpy_time,
        "jax_compile_seconds": compile_time,
        "jax_seconds": jax_time,
        "ok": max(forward, residual, chi2_error) <= rtol,
    }


def _compare_gradients(experiments, parameterization, frame, free, args) -> list[dict]:
    import chemex.jax as cj
    from chemex.parameters.program_evaluation import independent_dependencies

    usage = Counter(
        i
        for experiment in experiments
        for profile in experiment.profiles
        for i in independent_dependencies(parameterization, profile.name_map.values())
        if i in free
    )
    chosen = [i for i, _ in usage.most_common(args.gradients)]
    residuals = cj.compile_residuals(experiments, parameterization, chosen, frame=frame)
    x0 = residuals.x0
    jacobian = np.asarray(cj.jacobian(residuals, x0, chunk_size=args.chunk_size))
    _, evaluator = _native(experiments, parameterization, frame)
    from chemex.evaluation.native import EvaluationFrame

    def native_residuals(updates: dict[str, float]) -> np.ndarray | None:
        """Native residuals at shifted values; None outside the model's domain."""
        shifted = EvaluationFrame.from_lifecycle_frame(
            parameterization, frame.with_updates(updates)
        )
        result = evaluator.evaluate_residuals(shifted)
        return result if isinstance(result, np.ndarray) else None

    reference = native_residuals({})
    signal = float(np.max(np.abs(reference))) if reference is not None else 1.0
    rows = []
    for k, param_id in enumerate(chosen):
        column = jacobian[:, k]
        value = float(x0[k])
        scale = abs(value) if value else 1.0
        best = np.inf
        for step in FD_STEPS:
            h = step * scale
            points = [
                native_residuals({param_id: value + d}) for d in (h, -h, h / 2, -h / 2)
            ]
            plus, minus, plus_half, minus_half = points
            if plus is None or minus is None or plus_half is None or minus_half is None:
                continue
            fd = (4 * (plus_half - minus_half) / h - (plus - minus) / (2 * h)) / 3
            best = min(best, float(np.max(np.abs(column - fd))))
        column_max = float(np.max(np.abs(column)))
        effect = column_max * scale / signal
        rows.append(
            {
                "parameter": param_id,
                "effect": effect,
                # Zero derivative: measure the difference on the residual scale.
                "relative_error": best / (column_max or signal),
                "fd_limited": effect < FD_EFFECT_FLOOR,
            }
        )
    cj.release_memory()
    return rows


def _table(title: str) -> Any:
    from rich import box
    from rich.table import Table

    return Table(
        title=title,
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        pad_edge=False,
        collapse_padding=True,
        show_edge=False,
    )


def _print_report(reports: list[dict], gradients: list[dict], rtol: float) -> None:
    from chemex.messages import console

    table = _table("NumPy vs JAX backend")
    for column in (
        "Experiment",
        "Prof.",
        "Kern.",
        "Calc.",
        "Resid.",
        "χ²",
        "NumPy",
        "JAX",
        "Compile",
        "OK",
    ):
        table.add_column(column, no_wrap=True)
    for r in reports:
        table.add_row(
            Path(r["experiment"]).stem,
            str(r["profiles"]),
            str(r["kernels"]),
            f"{r['forward_error']:.0e}",
            f"{r['residual_error']:.0e}",
            f"{r['chi2_error']:.0e}",
            f"{r['numpy_seconds'] * 1e3:.0f} ms",
            f"{r['jax_seconds'] * 1e3:.0f} ms",
            f"{r['jax_compile_seconds']:.1f} s",
            "yes" if r["ok"] else "[red]NO[/red]",
        )
    console.print(table)
    console.print(
        f"Relative errors (Calc.: max|Δc|/max|c| over profiles; Resid.: "
        f"max|Δr|/max|s·c/e|; χ²: relative), tolerance {rtol:g}.  NumPy: all "
        "profiles' calculate_unscaled.  JAX: compiled calculation after the first "
        "call.  Compile: first call.  Kern.: compiled kernels (profiles sharing a "
        "layout share one)."
    )
    if gradients:
        grad_table = _table("Derivatives: JAX vs finite differences of NumPy")
        for column in ("Parameter", "Effect", "Relative difference", "Note"):
            grad_table.add_column(column)
        for g in gradients:
            note = "FD noise-limited" if g["fd_limited"] else ""
            grad_table.add_row(
                g["parameter"],
                f"{g['effect']:.1e}",
                f"{g['relative_error']:.1e}",
                note,
            )
        console.print(grad_table)
        console.print(
            "Effect = max|dr/dθ|·|θ| / max|r|.  Finite differences are limited by "
            "rounding noise (roughly 1e-12 / (effect x step)): differences for "
            f"parameters with an effect below {FD_EFFECT_FLOOR:g} reflect the "
            "finite-difference reference, not JAX."
        )


def compare_backends(args: Namespace) -> None:
    """Entry point of ``chemex compare-backends``."""
    from chemex.messages import error_console

    if find_spec("jax") is None:
        error_console.print(
            "chemex compare-backends needs the optional JAX backend: "
            "pip install 'chemex[jax]'",
            markup=False,
        )
        sys.exit(2)
    import chemex.jax  # noqa: F401 - enables float64 before anything else
    from chemex.parameters.parameterization import ParameterRole

    experiments, parameterization, frame = _build(args)
    free = [
        i
        for i in parameterization.independent_ids
        if parameterization.role(i) == ParameterRole.FIT
    ] or list(parameterization.independent_ids)
    reports = [
        _compare_experiment(e, parameterization, frame, free, args.rtol)
        for e in experiments
    ]
    gradients = (
        _compare_gradients(experiments, parameterization, frame, free, args)
        if args.gradients > 0
        else []
    )
    _print_report(reports, gradients, args.rtol)
    if args.json is not None:
        args.json.write_text(
            json.dumps({"experiments": reports, "gradients": gradients}, indent=2)
        )
    if not all(r["ok"] for r in reports):
        sys.exit(1)
