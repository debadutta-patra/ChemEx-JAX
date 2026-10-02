# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Write synthetic mini-examples for experiment types without a shipped example.

Each directory mirrors ``examples/<group>/<name>/`` (``run.sh``,
``Experiments/``, ``Data/``, ``Parameters/``) so the backend tests build it
exactly like a shipped example.  Settings follow the sample configurations in
``website/docs/experiments/**`` (or the module's settings class when no page
exists); spin systems and parameter values come from the closest shipped
example.  Intensities are placeholders: parity tests use only the metadata.

Run from the repository root:  uv run python tests/backend/synthetic/_generate.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

CEST_OFFSETS = np.concatenate([[-1.0e5], np.arange(-400.0, 401.0, 50.0)])
RUN_SH = """#!/bin/sh

chemex fit -e Experiments/*.toml \\
           -p Parameters/parameters.toml \\
           -d {model} \\
           -o Output
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"), encoding="utf-8")


def _columns(rows: list[tuple[object, ...]], header: str) -> str:
    lines = [header] + [
        " ".join(f"{v:>14}" if isinstance(v, str) else f"{v:14.6e}" for v in row)
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def _profiles_block(names: dict[str, str]) -> str:
    return "\n".join(f'{key} = "{value}"' for key, value in names.items())


def _example(
    name: str,
    experiment: str,
    profiles: dict[str, str],
    data: dict[str, str],
    parameters: str,
    *,
    model: str = "2st",
    data_path: str = "../Data/",
    data_extra: str = 'error = "file"',
) -> None:
    directory = ROOT / name
    if directory.exists():
        shutil.rmtree(directory)
    _write(directory / "run.sh", RUN_SH.format(model=model))
    _write(
        directory / "Experiments" / "experiment.toml",
        f"""
{experiment.strip()}

[data]
path = "{data_path}"
{data_extra}
{"[data.profiles]" if profiles else ""}
{_profiles_block(profiles)}
""",
    )
    for filename, text in data.items():
        _write(directory / "Data" / filename, text)
    _write(directory / "Parameters" / "parameters.toml", parameters)


def _cest_data(names: list[str]) -> dict[str, str]:
    rows = [(o, 1.0e5, 1.0e3) for o in CEST_OFFSETS]
    return {
        f"{n}.out": _columns(rows, "# Offset (Hz) Intensity Uncertainty") for n in names
    }


CARBON = ["L7CD2", "I28CD1", "V58CG2"]
CARBON_PARAMETERS = """
[GLOBAL]
PB = 0.045
KEX_AB = 50.0
R1_A = 2.3
R2_A = 15.0

[CS_A]
L7CD2 = 21.63866
I28CD1 = 10.07248
V58CG2 = 21.26392

[DW_AB]
L7CD2 = 1.5
I28CD1 = 2.5
V58CG2 = -1.0
"""

NITROGEN = ["13N", "43N", "55N"]
NITROGEN_PARAMETERS = """
[GLOBAL]
PB = 0.015
KEX_AB = 70.0
R1_A = 1.5
R2_A = 8.0

[CS_A]
13N = 108.207
43N = 108.876
55N = 128.301

[DW_AB]
13N = 4.0
43N = 12.5
55N = -6.5
"""

SHIFT_PARAMETERS = """
[GLOBAL]
PB = 0.02
KEX_AB = 4000.0
DW_AB = 0.05

[CS_A]
11N = 115.384
29N = 133.840
112N = 124.653
11C = 55.0
29C = 60.0
112C = 57.5

[DW_AB]
11N = 3.5
29N = -4.0
112N = -6.0
11C = 1.5
29C = -2.0
112C = 2.5
"""


def main() -> None:
    _example(
        "DCEST_13C",
        """
[experiment]
name = "dcest_13c"
time_t1 = 0.5
carrier = 20.0
pw90 = 15e-6
sw = 800.0
b1_eff = 25.0
b1_distribution = { type = "gaussian", scale = 0.1, res = 3 }

[conditions]
h_larmor_frq = 600.0
""",
        {n: f"{n}.out" for n in CARBON},
        _cest_data(CARBON),
        CARBON_PARAMETERS,
    )
    _example(
        "COSCEST_13C",
        """
[experiment]
name = "coscest_13c"
time_t1 = 0.3
carrier = 20.0
sw = 800.0
cos_n = 3
cos_res = 10
b1_frq = 20.0
b1_distribution = { type = "gaussian", scale = 0.1, res = 3 }

[conditions]
h_larmor_frq = 600.0
""",
        {n: f"{n}.out" for n in CARBON},
        _cest_data(CARBON),
        CARBON_PARAMETERS,
    )
    _example(
        "CEST_15N_TEST",
        """
[experiment]
name = "cest_15n_test"
time_t1 = 0.5
carrier = 118.987
b1_frq = 26.3
b1_inh_scale = 0.1
b1_inh_res = 3

[conditions]
h_larmor_frq = 499.243
""",
        {n: f"{n}.out" for n in NITROGEN},
        _cest_data(NITROGEN),
        NITROGEN_PARAMETERS,
    )
    times = [0.0, 0.01, 0.02, 0.04, 0.08]
    relaxation = {
        f"{n}.out": _columns(
            [(t, 1.0e5 * np.exp(-10.0 * t), 1.0e3) for t in times],
            "# Time (s) Intensity Uncertainty",
        )
        for n in NITROGEN
    }
    for name, module, distribution in (
        ("RELAXATION_15N_R1RHO", "wip.relaxation_15n_r1rho", "res = 3, scale = 0.1"),
        (
            "RELAXATION_15N_R1RHO_EIG",
            "wip.relaxation_15n_r1rho_eig",
            "res = 1, scale = 0.0",
        ),
    ):
        _example(
            name,
            f"""
[experiment]
name = "{module}"
carrier = 118.0
b1_frq = 1500.0
b1_distribution = {{ type = "gaussian", {distribution} }}

[conditions]
h_larmor_frq = 600.0
""",
            {n: f"{n}.out" for n in NITROGEN},
            relaxation,
            NITROGEN_PARAMETERS,
        )
    noesy_rows = [
        (t, s1, s2, 1.0e5, 1.0e3)
        for t in (0.0, 0.05, 0.1, 0.2)
        for s1 in "ab"
        for s2 in "ab"
    ]
    _example(
        "NOESYFPGPPH19",
        """
[experiment]
name = "noesyfpgpph19"

[conditions]
h_larmor_frq = 600.0
""",
        {n: f"{n}.out" for n in NITROGEN},
        {
            f"{n}.out": _columns(noesy_rows, "# Time State1 State2 Intensity Error")
            for n in NITROGEN
        },
        NITROGEN_PARAMETERS.replace("KEX_AB = 70.0", "KEX_AB = 20.0").replace(
            "PB = 0.015", "PB = 0.3"
        ),
    )
    ncycs = [0.0, 1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 30.0]
    _example(
        "MODELS",
        """
[experiment]
name = "cpmg_15n_ip"
carrier = 118.5
pw90 = 40.0e-6
time_equil = 2.0e-3
time_t2 = 30.0e-3

[conditions]
h_larmor_frq = 600.0
temperature = 25.0
p_total = 1.5e-3
l_total = 1.0e-3
d2o = 0.1
""",
        {n: f"{n}.out" for n in NITROGEN},
        {
            f"{n}.out": _columns(
                [(c, 1.0e5, 1.0e3) for c in ncycs], "# ncyc Intensity Uncertainty"
            )
            for n in NITROGEN
        },
        """
[CS_A]
13N = 108.207
43N = 108.876
55N = 128.301
""",
    )
    for name, module, suffix in (
        ("SHIFT_15N_SQ", "shift_15n_sq", "N"),
        ("SHIFT_13C_SQ", "shift_13c_sq", "C"),
    ):
        spins = [f"{r}{suffix}" for r in (11, 29, 112)]
        _example(
            name,
            f"""
[experiment]
name = "{module}"

[conditions]
h_larmor_frq = 499.684
""",
            {},
            {
                "shifts.txt": _columns(
                    [(s, 1.0, 0.5) for s in spins], "# Name Shift Error"
                )
            },
            SHIFT_PARAMETERS,
            data_path="../Data/shifts.txt",
            data_extra="",
        )


if __name__ == "__main__":
    main()
