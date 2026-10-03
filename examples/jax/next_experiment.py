# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Choose the next experiment given data you already have (sequential design).

Fisher information adds over independent experiments:

    F_next = F_existing + F_candidate

* ``F_existing`` = JᵀJ of ChemEx's native weighted residuals of the data you
  have (real errors, per-profile scale profiled out exactly as in the fit),
  evaluated at your current estimates (ideally your fitted values).
* For each candidate setting, the script writes a copy of your experiment TOML
  with that one setting changed, builds it *jointly* with the existing data
  through ChemEx's normal setup (so new field-dependent nuisance parameters,
  e.g. R2 at a new B0, are included), and fills its data with ChemEx's own
  noiseless prediction at the current estimates, with errors equal to the
  existing data's relative noise.  ``F_candidate`` is then JᵀJ of the
  candidate's native weighted residuals.

The candidate that most reduces the predicted standard errors of the
parameters of interest is the most informative next experiment.

Modes (run from the repository root):

    # next B0 field for a 15N CPMG data set recorded at 500 MHz
    uv run --extra jax python examples/jax/next_experiment.py cpmg-b0
    # next B1 field for a 15N CEST data set recorded with B1 = 26 Hz
    uv run --extra jax python examples/jax/next_experiment.py cest-b1

Pass ``--parameters`` with your fitted values for a real decision; the
defaults use the examples' parameter files.  ``--snr-exponent k`` scales the
candidate's relative noise by ``(B0_existing / B0_candidate)**k`` (e.g. 1.5
for sensitivity growing as B0^1.5; default 0: same relative noise).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import jax
import numpy as np

import chemex.jax as cj  # enables float64; must precede other JAX use
from chemex.configuration.methods import Method, Selection
from chemex.configuration.parameters import read_defaults
from chemex.containers.experiments import Experiments
from chemex.experiments.builder import build_experiments
from chemex.parameters.parameterization import ParameterRole
from chemex.parameters.program_evaluation import independent_dependencies
from chemex.parameters.spin_system import SpinSystem
from chemex.runtime import AnalysisSession, ensure_plugins_registered

EXAMPLES = Path(__file__).resolve().parents[1] / "Experiments"
MODES: dict[str, dict[str, Any]] = {
    "cpmg-b0": {
        "example": "CPMG_15N_IP",
        "existing": "500mhz.toml",
        "section": "conditions",
        "key": "h_larmor_frq",
        "values": [600.0, 700.0, 800.0, 950.0, 1200.0],
        "residues": ["15", "31", "33", "34", "37"],
        "label": "B0 (1H MHz)",
    },
    "cest-b1": {
        "example": "CEST_15N",
        "existing": "26hz.toml",
        "section": "experiment",
        "key": "b1_frq",
        "values": [5.0, 10.0, 13.0, 20.0, 40.0],
        "residues": ["13", "43", "55"],
        "label": "B1 (Hz)",
    },
}


# --- minimal TOML writer for experiment files (scalars, lists, inline tables)
def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
        )
    msg = f"unsupported TOML value {value!r}"
    raise TypeError(msg)


def _write_toml(path: Path, document: dict[str, Any]) -> None:
    lines: list[str] = []

    def table(name: str, content: dict[str, Any]) -> None:
        lines.append(f"[{name}]")
        nested = {
            k: v for k, v in content.items() if isinstance(v, dict) and k == "profiles"
        }
        for key, value in content.items():
            if key not in nested:
                lines.append(f'"{key}" = {_toml_value(value)}')
        lines.append("")
        for key, value in nested.items():
            table(f"{name}.{key}", value)

    for name, content in document.items():
        table(name, content)
    path.write_text("\n".join(lines), encoding="utf-8")


def candidate_toml(
    template: Path,
    directory: Path,
    section: str,
    key: str,
    value: float,
    profiles: dict,
) -> Path:
    """Copy of ``template`` with ``[section] key = value`` and placeholder data."""
    document = tomllib.loads(template.read_text(encoding="utf-8"))
    document[section][key] = value
    data = document["data"]
    data_dir = directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data["path"] = "data/"
    data["error"] = "file"
    data["profiles"] = {}
    for name, metadata in profiles.items():
        rows = "\n".join(f"{m:.6e} 1.0 1.0" for m in metadata)
        (data_dir / f"{name}.out").write_text(rows + "\n", encoding="utf-8")
        data["profiles"][name] = f"{name}.out"
    path = directory / f"candidate_{key}_{value:g}.toml"
    _write_toml(path, document)
    return path


def build(files: list[Path], parameters: list[Path], residues: list[str]):
    """Build all files jointly (all residues, then select, as chemex fit does)."""
    ensure_plugins_registered()
    with contextlib.redirect_stdout(io.StringIO()):
        session = AnalysisSession()
        session.set_model("2st")
        experiments = build_experiments(
            files, Selection(include=None, exclude=None), session=session
        )
        experiments.select(
            Selection(include=[SpinSystem.from_name(r) for r in residues], exclude=None)
        )
        session.parameters.set_defaults(read_defaults(parameters))
        if not session.try_build_analysis_values():
            msg = "could not build analysis values"
            raise RuntimeError(msg)
    parameterization = session.compile_parameterization(Method(), experiments.param_ids)
    frame = parameterization.frame_from_snapshot(session.analysis_values.snapshot())
    return experiments, parameterization, frame


def _subset(experiments: Experiments, keep) -> Experiments:
    subset = Experiments()
    for experiment in experiments:
        if keep(experiment):
            subset.add(experiment)
    return subset


_FIELD = re.compile(r"_\d+_\d+MHZ$")


def _initialise_new_field_parameters(parameterization, frame):
    """Give parameters that exist only at the candidate field (e.g. R2 at a new
    B0) the value of the same parameter at the existing field."""
    values = dict(frame._items)
    by_stem: dict[str, float] = {}
    for param_id, value in values.items():
        if _FIELD.search(param_id):
            by_stem.setdefault(_FIELD.sub("", param_id), value)
    updates = {}
    for param_id in parameterization.independent_ids:
        stem = _FIELD.sub("", param_id)
        if (
            _FIELD.search(param_id)
            and stem in by_stem
            and values[param_id] != by_stem[stem]
        ):
            updates[param_id] = by_stem[stem]
    return frame.with_updates(updates) if updates else frame


def _fisher(experiments, parameterization, free, frame) -> np.ndarray:
    residuals = cj.compile_residuals(experiments, parameterization, free, frame=frame)
    jacobian = np.asarray(jax.jit(jax.jacfwd(residuals))(residuals.x0))
    return jacobian.T @ jacobian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--values", type=float, nargs="+", help="candidate settings")
    parser.add_argument("--residues", nargs="+")
    parser.add_argument("--parameters", type=Path, nargs="+")
    parser.add_argument("--snr-exponent", type=float, default=0.0)
    args = parser.parse_args()

    mode = MODES[args.mode]
    example = EXAMPLES / mode["example"]
    existing = example / "Experiments" / mode["existing"]
    parameters = args.parameters or sorted((example / "Parameters").glob("*.toml"))
    residues = args.residues or mode["residues"]
    interest = ("__KEX_AB", "__PB")

    # Existing data alone: its information and relative noise level.
    base, base_par, base_frame = build([existing], parameters, residues)
    base_free = [
        i for i in base_par.independent_ids if base_par.role(i) == ParameterRole.FIT
    ]
    f_base = _fisher(base, base_par, base_free, base_frame)
    base_se = np.sqrt(np.diag(np.linalg.inv(f_base)))
    relative_noise = float(
        np.median(
            [
                np.median(p.data.err[p.data.mask])
                / np.max(np.abs(p.data.exp[p.data.mask]))
                for e in base
                for p in e.profiles
            ]
        )
    )
    existing_value = tomllib.loads(existing.read_text())[mode["section"]][mode["key"]]
    values = dict(base_frame._items)
    print(
        f"{mode['example']}: existing {mode['label']} = {existing_value:g}, residues "
        f"{', '.join(residues)}, relative noise {relative_noise:.2%}"
    )
    print(f"| Next {mode['label']} | SE(KEX_AB) | SE(PB) |")
    print("| --- | --- | --- |")
    print(
        "| none (existing data only) | "
        + " | ".join(
            f"{base_se[base_free.index(i)] / abs(values[i]):.1%}" for i in interest
        )
        + " |"
    )

    template_profiles = {
        str(p.spin_system): np.asarray(p.data.metadata)
        for e in base
        for p in e.profiles
    }
    template = tomllib.loads(existing.read_text())
    names = {str(SpinSystem.from_name(k)): k for k in template["data"]["profiles"]}
    profiles = {names[s]: m for s, m in template_profiles.items() if s in names}

    for value in args.values or mode["values"]:
        with tempfile.TemporaryDirectory() as directory:
            candidate = candidate_toml(
                existing, Path(directory), mode["section"], mode["key"], value, profiles
            )
            joint, par, frame = build([existing, candidate], parameters, residues)
            frame = _initialise_new_field_parameters(par, frame)
            resolved = par.resolve(frame)
            name = candidate.name

            def is_candidate(e, name=name) -> bool:
                return Path(e.filename).name == name

            noise = relative_noise
            if mode["key"] == "h_larmor_frq":
                noise *= (existing_value / value) ** args.snr_exponent
            # Candidate data: ChemEx's noiseless prediction at the current
            # estimates, errors = the existing relative noise.
            for experiment in joint:
                if not is_candidate(experiment):
                    continue
                for profile in experiment.profiles:
                    calc = np.asarray(profile.calculate_unscaled(resolved), float)
                    reference = float(np.max(np.abs(calc)))
                    profile.data.exp = calc / reference
                    profile.data.err = np.full_like(calc, noise)
                    profile.data.mark_dirty()
            targets = [
                i
                for e in joint
                for p in e.profiles
                for i in independent_dependencies(par, p.name_map.values())
            ]
            free = [
                i for i in dict.fromkeys(targets) if par.role(i) == ParameterRole.FIT
            ]
            fisher = _fisher(
                _subset(joint, lambda e, keep=is_candidate: not keep(e)),
                par,
                free,
                frame,
            ) + _fisher(_subset(joint, is_candidate), par, free, frame)
            se = np.sqrt(np.diag(np.linalg.inv(fisher)))
            values = dict(frame._items)
            print(
                f"| {value:g} | "
                + " | ".join(
                    f"{se[free.index(i)] / abs(values[i]):.1%}" for i in interest
                )
                + " |",
                flush=True,
            )
            cj.release_memory()


if __name__ == "__main__":
    main()
