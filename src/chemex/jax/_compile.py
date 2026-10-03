# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Compile ChemEx profiles and residuals into pure JAX functions.

Profiles whose numerical kernel is identical (same pulse-sequence settings,
observation metadata, local parameter names and spectrometer numerics; only
the spin-system label and the parameter values differ) form a *group*.  A
group is traced once, as ``vmap`` over its profiles' local parameter vectors,
and the compiled function is cached per kernel signature, so repeated
``compile_*`` calls and many residues share one compilation.

Purity: traced code works on a private deep copy of a template spectrometer
made inside each trace; the user's profiles, spectrometers and ChemEx caches
never see a tracer.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from chemex.backend import get_backend
from chemex.containers.data import Data
from chemex.containers.experiments import Experiments
from chemex.containers.profile import Profile
from chemex.parameters.parameterization import (
    ActiveParameterization,
    IndependentValueFrame,
    ResolvedParameterValues,
)
from chemex.parameters.program_evaluation import (
    evaluate_program,
    independent_dependencies,
    validate_concrete,
)

_JAX = get_backend("jax")
_CACHE_SIZE = 64


# ---------------------------------------------------------------------------
# Kernel signatures and the per-signature compilation cache
# ---------------------------------------------------------------------------
def _digest(update: Callable[[Any], None]) -> str:
    h = hashlib.sha256()
    update(h)
    return h.hexdigest()


def _array_bytes(h: Any, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())


def kernel_signature(profile: Profile) -> str:
    """Fingerprint of everything that determines a profile's computation.

    Excludes the spin-system label and the parameter values: two profiles
    with equal signatures compute the same function of their local values.
    """
    engine = profile.spectrometer._engine
    settings = getattr(profile.pulse_sequence, "settings", None)
    settings_json = (
        json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
        if hasattr(settings, "model_dump")
        else repr(settings)
    )

    def update(h: Any) -> None:
        h.update(type(profile.pulse_sequence).__module__.encode())
        h.update(type(profile.pulse_sequence).__qualname__.encode())
        h.update(settings_json.encode())
        h.update(json.dumps(sorted(profile.name_map)).encode())
        _array_bytes(h, profile.data.metadata)
        basis = engine.basis
        h.update(repr((basis.type, basis.spin_system, basis.extension)).encode())
        h.update(repr(basis.model.states).encode())
        for name in sorted(engine._matrices):
            h.update(name.encode())
            _array_bytes(h, engine._matrices[name])
        for value in (
            engine.ppm_i,
            engine.ppm_s,
            engine.carrier_i,
            engine.carrier_s,
            engine.offset_i,
            engine.offset_s,
            engine.b1_s,
            engine.gradient_dephasing,
            engine.h_frq,
        ):
            h.update(repr(float(value)).encode())
        b1 = engine._b1_i_state
        jeff = engine._jeff_i_state
        for array in (
            b1.axis.values,
            b1.axis.weights,
            b1.l_x,
            b1.l_y,
            jeff.axis.values,
            jeff.axis.weights,
            jeff.liouvillian,
            engine.l_b1x_s,
            engine.l_b1y_s,
            engine._readout._detect_vector,
        ):
            _array_bytes(h, array)
        h.update(repr(bool(b1.axis.distribution.dephasing)).encode())

    return _digest(update)


@dataclass
class _GroupKernel:
    """One compiled, vmapped kernel shared by all profiles with a signature."""

    names: tuple[str, ...]
    function: Callable[[jax.Array], jax.Array]


class _KernelCache:
    """Thread-safe bounded cache of compiled group kernels."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[str, _GroupKernel] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, signature: str, profile: Profile) -> _GroupKernel:
        with self._lock:
            kernel = self._items.get(signature)
            if kernel is not None:
                self._items.move_to_end(signature)
                return kernel
            kernel = _build_group_kernel(profile)
            self._items[signature] = kernel
            if len(self._items) > self._size:
                self._items.popitem(last=False)
            return kernel

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


KERNEL_CACHE = _KernelCache(_CACHE_SIZE)


def _build_group_kernel(profile: Profile) -> _GroupKernel:
    template = deepcopy(profile.spectrometer)
    sequence = deepcopy(profile.pulse_sequence)
    names = tuple(profile.name_map)
    metadata = np.array(profile.data.metadata, copy=True)
    exp_shape = np.shape(profile.data.exp)
    err_shape = np.shape(profile.data.err)

    def single(local: jax.Array) -> jax.Array:
        # A private copy per trace: tracers live and die with this object.
        spectrometer = template.with_backend(_JAX)
        # Traced local values in place of floats (the engine is backend-generic).
        values = cast("dict[str, float]", {n: local[i] for i, n in enumerate(names)})
        spectrometer.update(values)
        data = Data(
            exp=np.zeros(exp_shape),
            err=np.zeros(err_shape),
            metadata=np.array(metadata, copy=True),
        )
        return jnp.asarray(sequence.calculate(spectrometer, data))

    return _GroupKernel(names, jax.jit(jax.vmap(single)))


def _group_profiles(
    profiles: Sequence[Profile],
) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    for index, profile in enumerate(profiles):
        groups.setdefault(kernel_signature(profile), []).append(index)
    return list(groups.items())


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _as_float64(x: Any, size: int) -> jax.Array:
    x = jnp.asarray(x)
    if x.dtype != jnp.float64:
        msg = (
            f"chemex.jax functions need float64 inputs, got {x.dtype}; float32 "
            "gives falsely small uncertainties in fast exchange."
        )
        raise TypeError(msg)
    if x.shape != (size,):
        msg = f"expected a vector of {size} free parameter values, got shape {x.shape}"
        raise ValueError(msg)
    return x


@dataclass
class _Program:
    parameterization: ActiveParameterization
    frame: IndependentValueFrame
    free_ids: tuple[str, ...]
    targets: tuple[str, ...]
    fixed: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        independent = set(self.parameterization.independent_ids)
        unknown = [i for i in self.free_ids if i not in independent]
        if unknown:
            msg = f"free_ids must be independent parameters; not independent: {unknown}"
            raise ValueError(msg)
        if len(set(self.free_ids)) != len(self.free_ids):
            raise ValueError("free_ids must be unique")
        needed = independent_dependencies(self.parameterization, self.targets)
        self.fixed = {
            i: v for i, v in self.frame._items if i in needed and i not in self.free_ids
        }

    @property
    def x0(self) -> np.ndarray:
        values = dict(self.frame._items)
        return np.array([values[i] for i in self.free_ids], dtype=np.float64)

    def resolve(self, x: jax.Array) -> dict[str, Any]:
        independent: dict[str, Any] = dict(self.fixed)
        independent.update({i: x[k] for k, i in enumerate(self.free_ids)})
        return evaluate_program(
            self.parameterization, independent, _JAX, targets=self.targets
        )

    def validate(self, x: Iterable[float]) -> ResolvedParameterValues:
        updates = dict(zip(self.free_ids, map(float, x), strict=True))
        return validate_concrete(self.parameterization, self.frame, updates)


def _local_matrix(
    values: dict[str, Any], profiles: Sequence[Profile], names: tuple[str, ...]
) -> jax.Array:
    return jnp.stack(
        [
            jnp.stack([jnp.asarray(values[p.name_map[n]]) for n in names])
            for p in profiles
        ]
    )


# ---------------------------------------------------------------------------
# Public compiled objects
# ---------------------------------------------------------------------------
@dataclass(eq=False)  # identity semantics: hashable for jax.jit
class CompiledProfile:
    """``f(x) -> unscaled profile`` for one ChemEx profile.

    ``x`` holds the values of ``free_ids`` (float64); other independent
    parameters are fixed at ``frame``.  Pure and traceable (``jit``, ``vmap``,
    ``grad``).  Call :meth:`validate` on concrete inputs to run ChemEx's
    domain checks, which cannot run on traced values.
    """

    free_ids: tuple[str, ...]
    _program: _Program = field(repr=False)
    _profile: Profile = field(repr=False)
    _kernel: _GroupKernel = field(repr=False)

    @property
    def x0(self) -> np.ndarray:
        """Free parameter values from ``frame``."""
        return self._program.x0

    def __call__(self, x: Any) -> jax.Array:
        x = _as_float64(x, len(self.free_ids))
        values = self._program.resolve(x)
        local = _local_matrix(values, [self._profile], self._kernel.names)
        return self._kernel.function(local)[0]

    def validate(self, x: Iterable[float]) -> ResolvedParameterValues:
        """Run ChemEx's checked resolver on concrete ``x`` (raises on domain errors)."""
        return self._program.validate(x)


@dataclass
class _ResidualGroup:
    kernel: _GroupKernel
    profiles: list[Profile]
    exp: np.ndarray
    err: np.ndarray
    mask: np.ndarray
    scaled: np.ndarray


@dataclass(eq=False)  # identity semantics: hashable for jax.jit
class CompiledResiduals:
    """``r(x)`` = ChemEx's native weighted residual vector.

    Same order (experiment, profile, retained observation), same per-profile
    normalisation (``Σ(c/e)(x/e) / Σ(c/e)²`` for scaled profiles, else 1) and
    same weighting ``(scale·calc − exp) / err`` as
    ``chemex.evaluation.native.EvaluationEngine``.  ``chi2(x) = Σ r²``.
    """

    free_ids: tuple[str, ...]
    _program: _Program = field(repr=False)
    _groups: list[_ResidualGroup] = field(repr=False)
    _order: np.ndarray = field(repr=False)
    _observations: np.ndarray = field(repr=False)

    @property
    def x0(self) -> np.ndarray:
        return self._program.x0

    @property
    def size(self) -> int:
        """Number of residuals (retained observations)."""
        return int(self._order.size)

    @property
    def group_count(self) -> int:
        """Number of distinct compiled kernels (shape/settings groups)."""
        return len(self._groups)

    def _calculations(self, x: Any) -> list[jax.Array]:
        x = _as_float64(x, len(self.free_ids))
        values = self._program.resolve(x)
        return [
            group.kernel.function(
                _local_matrix(values, group.profiles, group.kernel.names)
            )
            for group in self._groups
        ]

    def calculations(self, x: Any) -> jax.Array:
        """Unscaled calculated intensities of every observation (masked ones
        included), concatenated in ChemEx's experiment/profile/point order —
        the native evaluator's ``unscaled_calculations``."""
        flat = [calc.ravel() for calc in self._calculations(x)]
        return jnp.concatenate(flat)[self._observations]

    def __call__(self, x: Any) -> jax.Array:
        flat = []
        for group, calc in zip(self._groups, self._calculations(x), strict=True):
            mask = jnp.asarray(group.mask)
            err = jnp.asarray(group.err)
            exp = jnp.asarray(group.exp)
            weighted = jnp.where(mask, calc / err, 0.0)
            numerator = jnp.sum(weighted * jnp.where(mask, exp / err, 0.0), axis=1)
            denominator = jnp.sum(weighted * weighted, axis=1)
            scale = jnp.where(jnp.asarray(group.scaled), numerator / denominator, 1.0)
            residuals = jnp.where(mask, (scale[:, None] * calc - exp) / err, 0.0)
            flat.append(residuals.ravel())
        return jnp.concatenate(flat)[self._order]

    def chi2(self, x: Any) -> jax.Array:
        r = self(x)
        return jnp.sum(r * r)

    def validate(self, x: Iterable[float]) -> ResolvedParameterValues:
        """Run ChemEx's checked resolver on concrete ``x`` (raises on domain errors)."""
        return self._program.validate(x)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def compile_profile(
    profile: Profile,
    parameterization: ActiveParameterization,
    free_ids: Sequence[str],
    *,
    frame: IndependentValueFrame,
) -> CompiledProfile:
    """Compile one profile as a pure function of the ``free_ids`` values."""
    targets = tuple(profile.name_map.values())
    program = _Program(parameterization, frame, tuple(free_ids), targets)
    kernel = KERNEL_CACHE.get(kernel_signature(profile), profile)
    return CompiledProfile(tuple(free_ids), program, profile, kernel)


def _profile_arrays(profile: Profile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exp = np.asarray(profile.data.exp, dtype=np.float64)
    err = np.asarray(profile.data.err, dtype=np.float64)
    mask = np.asarray(profile.data.mask, dtype=np.bool_)
    if exp.ndim != 1 or err.shape != exp.shape or mask.shape != exp.shape:
        msg = "residuals need one-dimensional symmetric data (as native evaluation)"
        raise ValueError(msg)
    return exp, np.where(mask, err, 1.0), mask


def compile_residuals(
    experiments: Experiments,
    parameterization: ActiveParameterization,
    free_ids: Sequence[str],
    *,
    frame: IndependentValueFrame,
) -> CompiledResiduals:
    """Compile the native weighted residual vector of ``experiments``."""
    profiles = [
        profile for experiment in experiments for profile in experiment.profiles
    ]
    targets = tuple(dict.fromkeys(pid for p in profiles for pid in p.name_map.values()))
    program = _Program(parameterization, frame, tuple(free_ids), targets)

    groups: list[_ResidualGroup] = []
    position: dict[int, tuple[int, int]] = {}
    offsets: list[int] = []
    offset = 0
    for g, (signature, indices) in enumerate(_group_profiles(profiles)):
        members = [profiles[i] for i in indices]
        arrays = [_profile_arrays(p) for p in members]
        groups.append(
            _ResidualGroup(
                kernel=KERNEL_CACHE.get(signature, members[0]),
                profiles=members,
                exp=np.stack([np.where(m, e, 0.0) for e, _, m in arrays]),
                err=np.stack([e for _, e, _ in arrays]),
                mask=np.stack([m for _, _, m in arrays]),
                scaled=np.array([p.is_scaled for p in members]),
            )
        )
        for row, i in enumerate(indices):
            position[i] = (g, row)
        offsets.append(offset)
        offset += len(indices) * arrays[0][0].size

    order: list[int] = []
    observations: list[int] = []
    for i, profile in enumerate(profiles):
        g, row = position[i]
        n_points = groups[g].exp.shape[1]
        start = offsets[g] + row * n_points
        order.extend(start + np.flatnonzero(np.asarray(profile.data.mask)))
        observations.extend(start + np.arange(n_points))
    return CompiledResiduals(
        tuple(free_ids),
        program,
        groups,
        np.asarray(order, dtype=np.intp),
        np.asarray(observations, dtype=np.intp),
    )


def release_memory() -> None:
    """Drop compiled kernels and return freed heap memory to the OS.

    Clears this module's kernel cache and JAX's compilation caches, runs the
    garbage collector and, on glibc systems, ``malloc_trim(0)``.  Without the
    trim a long-lived process keeps its high-water mark (measured: 4.4 GB
    stays 4.4 GB after clearing; with the trim it drops to 0.7-2.3 GB).
    Compiled functions created before the call recompile on next use.
    """
    import ctypes
    import ctypes.util
    import gc

    KERNEL_CACHE.clear()
    jax.clear_caches()
    gc.collect()
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    try:
        trim = ctypes.CDLL(libc_name).malloc_trim
    except (OSError, AttributeError):  # not glibc
        return
    trim(0)


def _in_chunks(
    apply: Callable[[jax.Array], jax.Array], size: int, chunk_size: int, dtype: Any
) -> jax.Array:
    """Stack ``apply(basis_block)`` over identity blocks of ``chunk_size`` rows.

    The last block is zero-padded to the same shape so ``apply`` compiles once.
    """
    blocks = []
    for start in range(0, size, chunk_size):
        block = jnp.eye(size, dtype=dtype)[start : start + chunk_size]
        used = block.shape[0]
        if used < chunk_size:
            padding = jnp.zeros((chunk_size - used, size), dtype)
            block = jnp.concatenate([block, padding])
        blocks.append(apply(block)[:used])
    return jnp.concatenate(blocks, axis=0)


def jacobian(
    function: Callable[[jax.Array], jax.Array],
    x: Any,
    *,
    chunk_size: int | None = None,
    mode: str = "forward",
) -> jax.Array:
    """Jacobian of ``function`` at ``x``, optionally in memory-bounded chunks.

    ``mode="forward"`` builds columns (one per parameter) and suits few
    parameters / many residuals; ``mode="reverse"`` builds rows and suits many
    parameters / few outputs.  Without ``chunk_size`` this is exactly
    ``jax.jacfwd`` / ``jax.jacrev``.  With ``chunk_size = k``, at most ``k``
    columns (forward) or rows (reverse) are computed at once, which bounds the
    tangent/cotangent memory to ``k`` copies of the calculation instead of one
    per parameter (forward) or output (reverse); the chunk function is compiled
    once (the last chunk is zero-padded).  Forward chunks recompute the
    primal once per chunk; reverse chunks share one linearisation.
    """
    x = jnp.asarray(x)
    if mode not in ("forward", "reverse"):
        msg = f"mode must be 'forward' or 'reverse', got {mode!r}"
        raise ValueError(msg)
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if mode == "forward":
        if chunk_size is None or chunk_size >= x.shape[0]:
            return jax.jacfwd(function)(x)

        @jax.jit
        def columns(tangents: jax.Array) -> jax.Array:
            return jax.vmap(lambda t: jax.jvp(function, (x,), (t,))[1])(tangents)

        return _in_chunks(columns, x.shape[0], chunk_size, x.dtype).T

    output, pullback = jax.vjp(function, x)
    if chunk_size is None or chunk_size >= output.shape[0]:
        return jax.jacrev(function)(x)
    rows = jax.jit(jax.vmap(lambda c: pullback(c)[0]))
    return _in_chunks(rows, output.shape[0], chunk_size, output.dtype)
