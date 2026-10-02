# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Backend-generic evaluation of a compiled constraint program.

:func:`evaluate_program` walks ``ActiveParameterization``'s ordered
constraints with exactly the tree semantics of
``ActiveParameterization._resolve_values`` (same order, same nodes, same
``-0.0`` normalisation), but lets the array backend evaluate scientific
functions, so independent values may be JAX tracers.

Domain checks that need concrete values (finiteness, ``np.errstate``,
arithmetic exceptions, model-owned function domains) are *not* part of the
traced path.  :func:`validate_concrete` runs ChemEx's own resolver on concrete
values instead; callers run it on every concrete input they evaluate, so no
check is ever dropped.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any, cast

from chemex.backend import NUMPY_BACKEND, Backend
from chemex.parameters.parameterization import (
    ActiveParameterization,
    BinaryExpression,
    CompiledConstraint,
    FunctionExpression,
    IndependentValueFrame,
    LiteralExpression,
    ReferenceExpression,
    ResolvedParameterValues,
    ScalarExpression,
    UnaryExpression,
    UnsupportedConstraintExpressionError,
)

_BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


def required_constraints(
    parameterization: ActiveParameterization,
    targets: Iterable[str],
) -> tuple[CompiledConstraint, ...]:
    """Constraints needed to resolve ``targets``, in program evaluation order."""
    by_target = {c.target_id: c for c in parameterization.ordered_constraints}
    needed: set[str] = set()
    pending = [t for t in targets if t in by_target]
    while pending:
        target = pending.pop()
        if target in needed:
            continue
        needed.add(target)
        pending.extend(d for d in by_target[target].dependencies if d in by_target)
    return tuple(
        c for c in parameterization.ordered_constraints if c.target_id in needed
    )


def independent_dependencies(
    parameterization: ActiveParameterization,
    targets: Iterable[str],
) -> tuple[str, ...]:
    """Independent parameter ids that ``targets`` depend on (program order)."""
    targets = tuple(targets)
    independent = set(parameterization.independent_ids)
    used = {t for t in targets if t in independent}
    for constraint in required_constraints(parameterization, targets):
        used.update(d for d in constraint.dependencies if d in independent)
    return tuple(i for i in parameterization.independent_ids if i in used)


def _evaluate(
    node: ScalarExpression,
    values: Mapping[str, Any],
    parameterization: ActiveParameterization,
    backend: Backend,
) -> Any:
    if isinstance(node, LiteralExpression):
        return node.value
    if isinstance(node, ReferenceExpression):
        return values[node.param_id]
    if isinstance(node, UnaryExpression):
        operand = _evaluate(node.operand, values, parameterization, backend)
        return +operand if node.operator == "positive" else -operand
    if isinstance(node, BinaryExpression):
        left = _evaluate(node.left, values, parameterization, backend)
        right = _evaluate(node.right, values, parameterization, backend)
        return _BINARY[node.operator](left, right)
    if isinstance(node, FunctionExpression):
        arguments = tuple(
            _evaluate(argument, values, parameterization, backend)
            for argument in node.arguments
        )
        result = backend.evaluate_scientific_function(
            node.function_id, parameterization.binder[node.function_id], arguments
        )
        if node.component is None:
            return result
        return cast("Mapping[str, Any]", result)[node.component]
    raise UnsupportedConstraintExpressionError(
        "Unknown constraint expression node", node_type=type(node).__name__
    )


def evaluate_program(
    parameterization: ActiveParameterization,
    independent: Mapping[str, Any],
    backend: Backend = NUMPY_BACKEND,
    *,
    targets: Collection[str] | None = None,
) -> dict[str, Any]:
    """Resolve derived values from ``independent`` with ``backend``.

    ``independent`` must contain every independent id the evaluated
    constraints depend on.  With ``targets``, only the constraints needed for
    those ids are evaluated.  Independent values and constraint results get
    ``+ 0.0`` (maps ``-0.0`` to ``+0.0`` like ChemEx's ``_finite_scalar``;
    derivative 1).
    """
    constraints = (
        parameterization.ordered_constraints
        if targets is None
        else required_constraints(parameterization, targets)
    )
    # Independent inputs get the same -0.0 -> +0.0 normalisation as ChemEx's
    # resolver applies to frame values.
    values = {param_id: value + 0.0 for param_id, value in independent.items()}
    for constraint in constraints:
        result = _evaluate(constraint.expression, values, parameterization, backend)
        values[constraint.target_id] = result + 0.0
    return values


def validate_concrete(
    parameterization: ActiveParameterization,
    frame: IndependentValueFrame,
    updates: Mapping[str, float] | None = None,
) -> ResolvedParameterValues:
    """Run ChemEx's own checked resolver on concrete values (raises on domain errors)."""
    if updates:
        frame = frame.with_updates({k: float(v) for k, v in updates.items()})
    return parameterization.resolve(frame)
