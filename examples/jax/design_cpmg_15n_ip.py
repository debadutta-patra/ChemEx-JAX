# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Optimal design of a 15N CPMG relaxation-dispersion experiment with exact Jacobians.

Local (prior-guess) optimal design for one residue of the CPMG_15N_IP example
(500 MHz):

1. Candidate measurements: every ncyc from 1 to ``ν_max · T2`` (ν_CPMG up to
   ``--nu-max`` Hz) plus the ncyc = 0 reference, put on a copy of the
   example's profile.  No data are needed: only the metadata determine what
   ChemEx computes.
2. One exact Jacobian of the profile at the prior guess, augmented with the
   intensity-scale column (ChemEx always fits a per-profile scale), gives the
   Fisher information of *any* choice of points as a weighted sum of rows,
   ``F(n) = Jᵀ diag(n/σ²) J`` with n the number of repeats per candidate.
3. Greedy selection of ``--points`` measurements minimising the predicted
   variance of the parameters of interest (KEX_AB and PB here), compared with
   the example's actual 26-point ncyc list.
4. Robustness: the chosen designs are re-scored over random draws of the
   prior (±30% on KEX_AB and PB) with one ``vmap``-ed Jacobian.
5. A setting that changes the compiled program (the CPMG period T2) is
   compared by re-optimising for each candidate value.

Noise model: constant σ equal to ``--noise`` × the reference intensity (the
example's data have ≈ 0.8%).  Results are predictions for the assumed
parameters, not guarantees.

    uv run --with "jax>=0.11" python examples/jax/design_cpmg_15n_ip.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np

import chemex.jax as cj  # enables float64; must precede other JAX use
from chemex.configuration.methods import Method, Selection
from chemex.configuration.parameters import read_defaults
from chemex.containers.data import Data
from chemex.experiments.builder import build_experiments
from chemex.runtime import AnalysisSession, ensure_plugins_registered

EXAMPLE = Path(__file__).resolve().parents[1] / "Experiments" / "CPMG_15N_IP"


def build():
    ensure_plugins_registered()
    cwd = Path.cwd()
    os.chdir(EXAMPLE)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            session = AnalysisSession()
            session.set_model("2st")
            experiments = build_experiments(
                [Path("Experiments/500mhz.toml")],
                Selection(include=None, exclude=None),
                session=session,
            )
            session.parameters.set_defaults(
                read_defaults([Path("Parameters/parameters.toml")])
            )
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


def with_points(profile, metadata: np.ndarray, **settings: float):
    """Copy of ``profile`` computing ``metadata`` (and optionally new settings)."""
    candidate = deepcopy(profile)
    size = metadata.size
    candidate.data = Data(exp=np.ones(size), err=np.ones(size), metadata=metadata)
    if settings:
        sequence = candidate.pulse_sequence
        sequence.settings = sequence.settings.model_copy(update=settings)
    return candidate


class Design:
    """Fisher information of repeat counts over a fixed candidate set."""

    def __init__(self, compiled, theta: np.ndarray, noise: float) -> None:
        jacobian = np.asarray(jax.jacfwd(compiled)(theta))
        intensities = np.asarray(compiled(theta))
        # Last column: the per-profile intensity scale ChemEx always fits.
        self.rows = np.column_stack([jacobian, intensities])
        self.sigma = noise * intensities[0]  # ncyc = 0 reference is first

    def fisher(self, counts: np.ndarray) -> np.ndarray:
        return (self.rows.T * (counts / self.sigma**2)) @ self.rows

    def variances(self, counts: np.ndarray, interest: list[int]) -> np.ndarray:
        fisher = self.fisher(counts)
        if np.linalg.matrix_rank(fisher) < fisher.shape[0]:
            return np.full(len(interest), np.inf)
        return np.diag(np.linalg.inv(fisher))[interest]


def greedy(
    design: Design, n_points: int, interest: list[int], scale: np.ndarray
) -> np.ndarray:
    """Add one measurement at a time, minimising Σ relative variances of ``interest``."""
    counts = np.zeros(design.rows.shape[0])
    counts[0] = 2.0  # duplicate reference, as in the example
    # Seed with enough distinct points to make the information matrix regular.
    for index in (
        np.linspace(1, counts.size - 1, design.rows.shape[1]).round().astype(int)
    ):
        counts[index] += 1.0
    while counts.sum() < n_points:
        scores = []
        for k in range(counts.size):
            trial = counts.copy()
            trial[k] += 1.0
            scores.append(np.sum(design.variances(trial, interest) / scale**2))
        counts[int(np.argmin(scores))] += 1.0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--residue", default="15N")
    parser.add_argument("--points", type=int, default=26)
    parser.add_argument("--noise", type=float, default=0.008)
    parser.add_argument("--nu-max", type=float, default=1000.0)
    args = parser.parse_args()

    experiments, parameterization, frame = build()
    profile = next(
        p for e in experiments for p in e.profiles if str(p.spin_system) == args.residue
    )
    settings = profile.pulse_sequence.settings
    free = [
        "__KEX_AB",
        "__PB",
        f"__DW_AB_{args.residue}",
        next(
            i
            for i in parameterization.independent_ids
            if i.startswith(f"__R2_A_{args.residue}_")
        ),
    ]
    interest = [0, 1]  # KEX_AB, PB

    def candidates(t2: float) -> np.ndarray:
        return np.arange(0.0, np.floor(args.nu_max * t2) + 1.0)

    t2 = settings.time_t2
    compiled = cj.compile_profile(
        with_points(profile, candidates(t2)), parameterization, free, frame=frame
    )
    theta = compiled.x0
    design = Design(compiled, theta, args.noise)
    scale = np.abs(theta[interest])

    # The example's measured design, as repeat counts over the candidates.
    measured = np.zeros(design.rows.shape[0])
    for ncyc in profile.data.metadata:
        measured[int(ncyc)] += 1.0
    optimal = greedy(design, args.points, interest, scale)

    print(
        f"Residue {args.residue}, T2 = {t2 * 1e3:.0f} ms, noise {args.noise:.1%}, "
        f"prior: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(free, theta, strict=True))
    )
    for name, counts in (("example design", measured), ("greedy optimum", optimal)):
        se = np.sqrt(design.variances(counts, interest))
        chosen = ", ".join(
            f"{int(n)}" + (f"×{int(c)}" if c > 1 else "")
            for n, c in zip(candidates(t2), counts, strict=True)
            if c
        )
        print(
            f"  {name:15s} SE(KEX) {se[0]:7.2f} s⁻¹ ({se[0] / scale[0]:.1%})  "
            f"SE(PB) {se[1]:.2e} ({se[1] / scale[1]:.1%})  ncyc: {chosen}"
        )

    # Robustness over the prior: one vmap-ed Jacobian for all draws.
    rng = np.random.default_rng(1)
    draws = np.repeat(theta[None, :], 64, axis=0)
    draws[:, interest] *= np.exp(0.3 * rng.standard_normal((64, len(interest))))
    jacobians = np.asarray(jax.jit(jax.vmap(jax.jacfwd(compiled)))(draws))
    profiles = np.asarray(jax.jit(jax.vmap(compiled))(draws))
    for name, counts in (("example design", measured), ("greedy optimum", optimal)):
        relative = []
        for jac, intensities, draw in zip(jacobians, profiles, draws, strict=True):
            rows = np.column_stack([jac, intensities])
            fisher = (rows.T * (counts / (args.noise * intensities[0]) ** 2)) @ rows
            relative.append(np.sqrt(np.diag(np.linalg.inv(fisher))[0]) / draw[0])
        print(
            f"  {name:15s} SE(KEX)/KEX over 64 prior draws: median "
            f"{np.median(relative):.1%}, 90th percentile {np.percentile(relative, 90):.1%}"
        )

    # A setting that changes the compiled program: re-optimise per T2.
    print(
        "CPMG period T2 (ν_CPMG ≤ "
        f"{args.nu_max:.0f} Hz, {args.points} points, greedy optimum each):"
    )
    for t2_candidate in (0.020, 0.030, 0.040, 0.060):
        candidate = with_points(profile, candidates(t2_candidate), time_t2=t2_candidate)
        compiled_t2 = cj.compile_profile(candidate, parameterization, free, frame=frame)
        design_t2 = Design(compiled_t2, theta, args.noise)
        counts = greedy(design_t2, args.points, interest, scale)
        se = np.sqrt(design_t2.variances(counts, interest))
        print(
            f"  T2 = {t2_candidate * 1e3:4.0f} ms: SE(KEX) {se[0] / scale[0]:.1%}, "
            f"SE(PB) {se[1] / scale[1]:.1%}"
        )
    cj.release_memory()


if __name__ == "__main__":
    main()
