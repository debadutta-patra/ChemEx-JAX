# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Timing of the JAX backend vs ChemEx NumPy (exploratory; no performance gate).

Part 1 reproduces the prototype table: one 26-point CPMG_15N_IP profile with
6 free parameters (CS_A, DW_AB, KEX_AB, PB, R1_A, R2_A of residue 1N at
500 MHz) through ``chemex.jax.compile_profile``.

Part 2 times whole-example weighted residuals (``compile_residuals``, every
independent parameter free) against ChemEx's native evaluator, reporting the
first-call (compile) and steady-state times of the forward pass, and of the
Jacobian with respect to a fit-step-sized subset (the parameters the first
five residues depend on, at most 24).  Forward-mode memory grows with the
number of tangents times the traced intermediates: a full Jacobian over every
independent parameter of a CEST example exhausted 30 GB.  Peak RSS is printed.

    uv run --with "jax>=0.11" python benchmarks/jax_vs_numpy.py [EXAMPLE ...]

Record the hardware/software context with any saved output.
"""

from __future__ import annotations

import contextlib
import glob
import io
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

import chemex.jax as cj  # enables float64 first

import jax  # noqa: I001

from chemex.configuration.methods import Method, Selection
from chemex.configuration.parameters import read_defaults
from chemex.evaluation.native import EvaluationEngine, EvaluationFrame
from chemex.experiments.builder import build_experiments
from chemex.parameters.program_evaluation import independent_dependencies
from chemex.parameters.spin_system import SpinSystem
from chemex.runtime import AnalysisSession, ensure_plugins_registered

ROOT = Path(__file__).resolve().parents[1] / "examples" / "Experiments"


def build(name: str, model: str = "2st", include: list[str] | None = None):  # noqa: ANN201
    ensure_plugins_registered()
    cwd = Path.cwd()
    os.chdir(ROOT / name)
    try:
        selection = Selection(
            include=[SpinSystem.from_name(i) for i in include] if include else None,
            exclude=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            session = AnalysisSession()
            session.set_model(model)
            experiments = build_experiments(
                sorted(Path(p) for p in glob.glob("Experiments/*.toml")),
                selection,
                session=session,
            )
            session.parameters.set_defaults(
                read_defaults(sorted(Path(p) for p in glob.glob("Parameters/*.toml")))
            )
            if not session.try_build_analysis_values():
                raise RuntimeError(name)
        parameterization = session.compile_parameterization(
            Method(), experiments.param_ids
        )
        frame = parameterization.frame_from_snapshot(
            session.analysis_values.snapshot()
        )
    finally:
        os.chdir(cwd)
    return experiments, parameterization, frame


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def timed(function, *args, repeat: int = 50) -> tuple[float, float]:  # noqa: ANN001
    """(first call incl. compile, mean of ``repeat`` later calls) in seconds."""
    start = time.perf_counter()
    jax.block_until_ready(function(*args))
    first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repeat):
        jax.block_until_ready(function(*args))
    return first, (time.perf_counter() - start) / repeat


def part1() -> None:
    experiments, parameterization, frame = build("CPMG_15N_IP", include=["1N"])
    profile = next(p for e in experiments for p in e.profiles)
    wanted = ("__CS_A_1N", "__DW_AB_1N", "__KEX_AB", "__PB")
    rates = [i for i in parameterization.independent_ids if i.startswith(("__R1_A_1N", "__R2_A_1N")) and "500" in i]
    free = (*wanted, *rates)
    values = dict(parameterization.resolve(frame))
    compiled = cj.compile_profile(profile, parameterization, free, frame=frame)
    x0 = compiled.x0

    numpy_time = timed(lambda: profile.calculate_unscaled(values), repeat=200)[1]
    forward = timed(jax.jit(compiled), x0, repeat=200)
    with_jacobian = timed(
        jax.jit(lambda x: (compiled(x), jax.jacfwd(compiled)(x))), x0, repeat=200
    )
    rng = np.random.default_rng(0)
    batch = x0 * (1.0 + 0.01 * rng.standard_normal((256, x0.size)))
    batched = timed(jax.jit(jax.vmap(compiled)), batch, repeat=20)

    print(f"\nPart 1 — one {profile.data.exp.size}-point CPMG_15N_IP profile, "
          f"{len(free)} free parameters ({', '.join(free)})")
    print("| Operation | Time | First call (compile) |")
    print("| --- | --- | --- |")
    print(f"| ChemEx NumPy forward | {numpy_time * 1e3:.2f} ms | — |")
    print(f"| JAX jit forward | {forward[1] * 1e3:.2f} ms | {forward[0]:.1f} s |")
    print(f"| JAX forward + full jacfwd | {with_jacobian[1] * 1e3:.2f} ms | {with_jacobian[0]:.1f} s |")
    print(f"| JAX vmap batch of 256, per profile | {batched[1] / 256 * 1e3:.3f} ms | {batched[0]:.1f} s |")
    print(f"Peak RSS after part 1: {peak_rss_gb():.1f} GB")
    cj.release_memory()


def part2(names: list[str]) -> None:
    print("\nPart 2 — whole-example weighted residuals, all independent parameters free")
    print("| Example | Profiles | Kernels | Free (fwd / Jacobian) | Native residuals | JAX residuals (compile) | JAX Jacobian (compile) | Peak RSS |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in names:
        experiments, parameterization, frame = build(name)
        engine = EvaluationEngine.from_experiments(experiments, parameterization)
        evaluator = engine.new_evaluator()
        # The native evaluator caches results per frame: perturb a parameter
        # shared by every profile on each call so all profiles recompute.
        values = dict(frame._items)
        shared = next(
            (i for i in ("__KEX_AB", "__PB") if i in values),
            parameterization.independent_ids[0],
        )
        frames = [
            EvaluationFrame.from_lifecycle_frame(
                parameterization,
                frame.with_updates({shared: values[shared] * (1.0 + 1e-6 * (k + 1))}),
            )
            for k in range(6)
        ]
        evaluator.evaluate_residuals(frames[0])
        start = time.perf_counter()
        for eval_frame in frames[1:]:
            evaluator.evaluate_residuals(eval_frame)
        native = (time.perf_counter() - start) / (len(frames) - 1)
        free = parameterization.independent_ids
        residuals = cj.compile_residuals(experiments, parameterization, free, frame=frame)
        forward = timed(jax.jit(residuals), residuals.x0, repeat=10)

        profiles = [p for e in experiments for p in e.profiles]
        subset = tuple(
            dict.fromkeys(
                i
                for p in profiles[:5]
                for i in independent_dependencies(parameterization, p.name_map.values())
            )
        )[:24]
        partial = cj.compile_residuals(experiments, parameterization, subset, frame=frame)
        jacobian = timed(jax.jit(jax.jacfwd(partial)), partial.x0, repeat=3)
        print(
            f"| {name} | {len(profiles)} | {residuals.group_count} | "
            f"{len(free)} / {len(subset)} | {native * 1e3:.1f} ms | "
            f"{forward[1] * 1e3:.1f} ms ({forward[0]:.1f} s) | "
            f"{jacobian[1] * 1e3:.1f} ms ({jacobian[0]:.1f} s) | {peak_rss_gb():.1f} GB |",
            flush=True,
        )
        del residuals, partial
        cj.release_memory()


def main() -> None:
    print(
        f"# {platform.processor() or platform.machine()}, {os.cpu_count()} threads, "
        f"Python {platform.python_version()}, JAX {jax.__version__}, "
        f"devices {jax.devices()}"
    )
    part1()
    part2(sys.argv[1:] or ["CPMG_15N_IP", "CEST_15N", "DCEST_15N", "RELAXATION_NZ"])


if __name__ == "__main__":
    main()
