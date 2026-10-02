# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Fisher information of the CPMG_15N_IP example with exact JAX Jacobians.

Builds ``examples/Experiments/CPMG_15N_IP`` with ChemEx's own setup code (as
its ``run.sh`` does), restricted to the residues of the example method's
STEP1, compiles ChemEx's native weighted residuals with ``chemex.jax`` and
computes

    F = Jᵀ W J      (W = diag(1/σ²) is already in the weighted residuals)

for the parameters that STEP1 fits.  Prints the eigen-spectrum of F (small
eigenvalues = poorly identified parameter combinations), its condition
number, and the linearised standard errors sqrt(diag(F⁻¹)).

Run from the repository root (needs the ``jax`` extra):

    uv run --with "jax>=0.11" python examples/jax/fisher_cpmg_15n_ip.py
    # at ChemEx's fitted values, after running the example's run.sh:
    uv run --with "jax>=0.11" python examples/jax/fisher_cpmg_15n_ip.py \\
        --parameters examples/Experiments/CPMG_15N_IP/Parameters/parameters.toml \\
                     examples/Experiments/CPMG_15N_IP/Output/STEP1/Parameters/fitted.toml

The profile scale factor is the analytic per-profile normalisation of the
native fit; like ChemEx's own covariance, F is the information about the
model parameters at that profiled scale.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
from pathlib import Path

import jax
import numpy as np

import chemex.jax as cj  # enables float64; must precede other JAX use
from chemex.configuration.methods import Method, Selection
from chemex.configuration.parameters import read_defaults
from chemex.experiments.builder import build_experiments
from chemex.parameters.parameterization import ParameterRole
from chemex.parameters.spin_system import SpinSystem
from chemex.runtime import AnalysisSession, ensure_plugins_registered

EXAMPLE = Path(__file__).resolve().parents[1] / "Experiments" / "CPMG_15N_IP"
STEP1_RESIDUES = ("15", "31", "33", "34", "37")


def build(parameters: list[Path], residues: tuple[str, ...]):
    ensure_plugins_registered()
    parameters = [p.resolve() for p in parameters]
    cwd = Path.cwd()
    os.chdir(EXAMPLE)
    try:
        # Build every residue (as run.sh does: duplicate-based noise is
        # estimated over all profiles of an experiment), then restrict to the
        # method step's residues, as ChemEx's fit does.
        with contextlib.redirect_stdout(io.StringIO()):
            session = AnalysisSession()
            session.set_model("2st")
            experiments = build_experiments(
                sorted(Path("Experiments").glob("*.toml")),
                Selection(include=None, exclude=None),
                session=session,
            )
            experiments.select(
                Selection(
                    include=[SpinSystem.from_name(r) for r in residues], exclude=None
                )
            )
            session.parameters.set_defaults(read_defaults(parameters))
            if not session.try_build_analysis_values():
                msg = "could not build analysis values"
                raise RuntimeError(msg)
        parameterization = session.compile_parameterization(
            Method(), experiments.param_ids
        )
        frame = parameterization.frame_from_snapshot(session.analysis_values.snapshot())
    finally:
        os.chdir(cwd)
    return experiments, parameterization, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--parameters",
        type=Path,
        nargs="+",
        default=[EXAMPLE / "Parameters" / "parameters.toml"],
        help="parameter files, later ones override (like chemex -p)",
    )
    parser.add_argument("--include", nargs="+", default=list(STEP1_RESIDUES))
    args = parser.parse_args()

    experiments, parameterization, frame = build(args.parameters, tuple(args.include))
    free_ids = tuple(
        i
        for i in parameterization.independent_ids
        if parameterization.role(i) == ParameterRole.FIT
    )
    residuals = cj.compile_residuals(
        experiments, parameterization, free_ids, frame=frame
    )
    x0 = residuals.x0
    residuals.validate(x0)  # ChemEx's domain checks on the concrete point

    jacobian_fn = jax.jit(jax.jacfwd(residuals))
    start = time.perf_counter()
    jacobian = np.asarray(jacobian_fn(x0))
    compile_and_run = time.perf_counter() - start
    start = time.perf_counter()
    jacobian = np.asarray(jacobian_fn(x0))
    run = time.perf_counter() - start

    # Information about relative changes: F_ij scaled by |x_i x_j| makes the
    # spectrum comparable across parameters with different units.
    scale = np.where(x0 != 0.0, np.abs(x0), 1.0)
    fisher = jacobian.T @ jacobian
    fisher_relative = fisher * np.outer(scale, scale)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_relative)
    covariance = np.linalg.inv(fisher)
    chi2 = float(np.sum(np.asarray(residuals(x0)) ** 2))

    print(f"CPMG_15N_IP, residues {', '.join(args.include)}")
    print(
        f"{residuals.size} residuals, {len(free_ids)} free parameters, "
        f"{residuals.group_count} compiled kernel group(s), chi2 = {chi2:.2f}"
    )
    print(
        f"Jacobian: {compile_and_run:.2f} s first call (compile), {run * 1e3:.2f} ms after"
    )
    print("\nParameter                          value         std. error   rel.")
    for i, name in enumerate(free_ids):
        error = float(np.sqrt(covariance[i, i]))
        print(f"  {name:32s} {x0[i]: .5e}  {error: .3e}  {error / scale[i]:.2e}")
    print("\nEigenvalues of the relative Fisher information (ascending):")
    print("  " + " ".join(f"{v:.3e}" for v in eigenvalues))
    print(f"Condition number: {eigenvalues[-1] / eigenvalues[0]:.3e}")
    weakest = eigenvectors[:, 0]
    top = np.argsort(np.abs(weakest))[::-1][:4]
    print("Least-determined direction (largest components):")
    for i in top:
        print(f"  {weakest[i]: .3f} x {free_ids[i]}")


if __name__ == "__main__":
    main()
