# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""``chemex compare-backends`` on a shipped example."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chemex.cli import build_parser
from tests.backend.conftest import HAS_JAX

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "Experiments" / "RELAXATION_NZ"
)


def _args(*extra: str):
    return build_parser().parse_args(
        [
            "compare-backends",
            "-e",
            str(EXAMPLE / "Experiments" / "800mhz.toml"),
            "-p",
            str(EXAMPLE / "Parameters" / "parameters.toml"),
            *extra,
        ]
    )


def test_parser_registers_subcommand_without_jax_import() -> None:
    args = _args("--gradients", "2", "--rtol", "1e-8")
    assert args.commands == "compare-backends"
    assert args.gradients == 2
    assert args.rtol == pytest.approx(1e-8)
    assert args.analysis_command is False


@pytest.mark.skipif(HAS_JAX, reason="checks the message shown without JAX")
def test_without_jax_exits_with_install_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args()
    with pytest.raises(SystemExit) as stop:
        args.func(args)
    assert stop.value.code == 2
    assert "chemex[jax]" in capsys.readouterr().err


@pytest.mark.jax
def test_compare_backends_reports_parity(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    args = _args("--gradients", "4", "--json", str(report))
    cwd = Path.cwd()
    os.chdir(EXAMPLE)
    try:
        args.func(args)  # exits only on a parity failure
    finally:
        os.chdir(cwd)
    content = json.loads(report.read_text())
    (experiment,) = content["experiments"]
    assert experiment["ok"] is True
    assert experiment["profiles"] == 5
    assert experiment["forward_error"] <= 1e-9
    assert experiment["residual_error"] <= 1e-9
    gradients = content["gradients"]
    assert len(gradients) == 4
    resolvable = [g for g in gradients if not g["fd_limited"]]
    assert resolvable  # the R1 rates of this example
    assert all(g["relative_error"] < 1e-5 for g in resolvable)
