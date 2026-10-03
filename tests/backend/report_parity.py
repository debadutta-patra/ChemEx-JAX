# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Print the JAX-vs-NumPy coverage tables used in JAX_PORT_NOTES.md.

    uv run --extra jax python -m tests.backend.report_parity [CASE...]

Runs the same checks as ``test_jax_parity.py`` (forward on 3 profiles,
gradients on the first profile) and prints one Markdown row per case, then a
per-experiment-type coverage table.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from tests.backend._checks import EFFECT_THRESHOLD, check_profile
from tests.backend._examples import all_cases, build_case


def main(cases: list[str]) -> None:
    by_type: dict[str, list[str]] = defaultdict(list)
    print(
        "| Case | Model | Types | Fwd (3 prof.) | Worst strong grad (FD / exact) | jacrev | Weak params (reported) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for case in cases:
        example = build_case(case)
        types = sorted({e.name for e in example.experiments})
        for name in types:
            by_type[name].append(case)
        forward = max(
            check_profile(p, example.values, gradients=False).forward_error
            for p in example.sample_profiles(3)
        )
        check = check_profile(example.profiles[0], example.values)
        strong = [g for g in check.gradients if g.strong]
        fd_ok = [g.error_vs_numpy for g in strong if g.error_vs_exact is None]
        exact = [g.error_vs_exact for g in strong if g.error_vs_exact is not None]
        grad = f"{max(fd_ok, default=0.0):.1e}"
        if exact:
            grad += f" / {max(exact):.1e} ({len(exact)} exact)"
        weak = [
            g for g in check.gradients if g.effect < EFFECT_THRESHOLD and g.effect > 0
        ]
        weak_text = (
            ", ".join(
                f"{g.name} {g.effect:.0e}: FD {g.error_vs_numpy:.0e}, JAX-FD {g.error_vs_jax_fd:.0e}"
                for g in weak
            )
            or "—"
        )
        status = "" if not check.failures() else " **FAIL**"
        print(
            f"| {case} | {example.model} | {', '.join(types)} | {forward:.1e} | "
            f"{grad}{status} | {check.jacrev_error:.1e} | {weak_text} |",
            flush=True,
        )
    print()
    print("| Experiment type | Covered by |")
    print("| --- | --- |")
    for name in sorted(by_type):
        print(f"| `{name}` | {', '.join(by_type[name])} |")


if __name__ == "__main__":
    main(sys.argv[1:] or all_cases())
