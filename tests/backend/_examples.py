# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Build shipped examples exactly as their ``run.sh`` would (test helper)."""

from __future__ import annotations

import contextlib
import glob
import io
import os
import shlex
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from chemex.configuration.methods import Method, Selection
from chemex.configuration.parameters import read_defaults
from chemex.containers.experiments import Experiments
from chemex.containers.profile import Profile
from chemex.experiments.builder import build_experiments
from chemex.parameters.parameterization import (
    ActiveParameterization,
    IndependentValueFrame,
    ResolvedParameterValues,
)
from chemex.parameters.spin_system import SpinSystem
from chemex.runtime import AnalysisSession, ensure_plugins_registered

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SYNTHETIC = Path(__file__).resolve().parent / "synthetic"


@dataclass(frozen=True)
class BuiltExample:
    name: str
    model: str
    experiments: Experiments
    parameterization: ActiveParameterization
    frame: IndependentValueFrame
    values: ResolvedParameterValues

    @property
    def profiles(self) -> list[Profile]:
        return [profile for experiment in self.experiments for profile in experiment]

    def sample_profiles(self, count: int = 3) -> list[Profile]:
        """Return ``count`` profiles spread over the example (first, middle, last)."""
        profiles = self.profiles
        if len(profiles) <= count:
            return profiles
        if count == 1:
            return profiles[:1]
        step = (len(profiles) - 1) / (count - 1)
        return [profiles[round(i * step)] for i in range(count)]


def parse_run_script(path: Path) -> tuple[list[str], list[str], str, list[str]]:
    text = path.read_text().replace("\\\n", " ")
    variables = {}
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.isidentifier() and not line.startswith(" "):
            variables[name] = value.strip().strip('"')
    line = next(line for line in text.splitlines() if "chemex fit" in line)
    for name, value in variables.items():
        line = line.replace(f"${name}", value)
    args = shlex.split(line)
    lists: dict[str, list[str]] = {"-e": [], "-p": [], "--include": []}
    aliases = {"--experiments": "-e", "--parameters": "-p"}
    model, i = "2st", 0
    while i < len(args):
        flag = aliases.get(args[i], args[i])
        if flag in lists:
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                lists[flag].append(args[i])
                i += 1
            continue
        if flag in ("-d", "--model"):
            model = args[i + 1]
            i += 2
            continue
        i += 1
    return lists["-e"], lists["-p"], model, lists["--include"]


@cache
def build_example(
    name: str,
    group: str = "Experiments",
    script: str = "run.sh",
    root: Path = EXAMPLES,
    model: str | None = None,
) -> BuiltExample:
    """Build experiments and resolved parameter values for one example.

    ``root`` is ``EXAMPLES`` (shipped) or ``SYNTHETIC`` (``group`` ignored);
    ``model`` overrides the model named by the run script.
    """
    ensure_plugins_registered()
    example = root / name if root == SYNTHETIC else root / group / name
    experiments_globs, parameter_globs, script_model, include = parse_run_script(
        example / script
    )
    model = model or script_model
    cwd = Path.cwd()
    os.chdir(example)
    try:
        exp_files = sorted({Path(p) for e in experiments_globs for p in glob.glob(e)})
        par_files = sorted({Path(p) for e in parameter_globs for p in glob.glob(e)})
        selection = Selection(
            include=[SpinSystem.from_name(i) for i in include] or None, exclude=None
        )
        with contextlib.redirect_stdout(io.StringIO()):
            session = AnalysisSession()
            session.set_model(model)
            experiments = build_experiments(exp_files, selection, session=session)
            session.parameters.set_defaults(read_defaults(par_files))
            if not session.try_build_analysis_values():
                msg = f"Could not build analysis values for {name}"
                raise RuntimeError(msg)
        snapshot = session.analysis_values.snapshot()
        parameterization = session.compile_parameterization(
            Method(), experiments.param_ids
        )
        frame = parameterization.frame_from_snapshot(snapshot)
        values = parameterization.resolve(frame)
    finally:
        os.chdir(cwd)
    return BuiltExample(name, model, experiments, parameterization, frame, values)


def example_runs() -> list[tuple[str, str, str]]:
    """``(group, name, script)`` for every shipped example run script.

    An example directory runs ``run.sh``; a directory without one (e.g.
    ``DCEST_15N_3States``) runs each of its ``run*.sh`` variants.
    """
    runs = []
    for directory in sorted(p for p in EXAMPLES.glob("*/*") if p.is_dir()):
        scripts = (
            [directory / "run.sh"]
            if (directory / "run.sh").exists()
            else sorted(directory.glob("run*.sh"))
        )
        runs += [(directory.parent.name, directory.name, s.name) for s in scripts]
    return runs


def synthetic_examples() -> list[str]:
    """Synthetic mini-examples for experiment types without a shipped example."""
    return sorted(
        p.parent.name for p in SYNTHETIC.glob("*/run.sh") if p.parent.name != "MODELS"
    )


def all_cases() -> list[str]:
    """Test ids: ``Experiments/X``, ``Combinations/X[/script]``, ``synthetic/X``."""
    cases = []
    for group, name, script in example_runs():
        stem = "" if script == "run.sh" else f"/{Path(script).stem}"
        cases.append(f"{group}/{name}{stem}")
    cases += [f"synthetic/{name}" for name in synthetic_examples()]
    return cases


def build_case(case: str) -> BuiltExample:
    """Build the example named by an :func:`all_cases` id."""
    parts = case.split("/")
    if parts[0] == "synthetic":
        return build_example(parts[1], root=SYNTHETIC)
    script = f"{parts[2]}.sh" if len(parts) > 2 else "run.sh"
    return build_example(parts[1], parts[0], script)


def model_cases() -> list[str]:
    """Every registered kinetic model, plain and with each extension."""
    from chemex.models.factory import model_factory
    from chemex.runtime import ensure_plugins_registered

    ensure_plugins_registered()
    return [
        f"{name}{extension}"
        for name in sorted(model_factory.setting_makers_registry)
        for extension in ("", ".mf", ".rs", ".tc")
    ]


def build_model_case(model: str) -> BuiltExample:
    """The synthetic CPMG 15N template built with ``model``."""
    return build_example("MODELS", root=SYNTHETIC, model=model)
