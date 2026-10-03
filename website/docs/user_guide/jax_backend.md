---
sidebar_position: 5
description: When and how to use the optional JAX backend of the ChemEx-JAX fork for exact derivatives, Fisher information, gradient-based sampling, and fast batch simulation.
---

# Using the JAX backend

:::note ChemEx-JAX fork
This page documents a feature of the **ChemEx-JAX fork** (GPL-3.0-or-later)
of ChemEx. Upstream ChemEx does not include the JAX backend.
:::

ChemEx computes every profile with NumPy. The fork adds an optional second
backend, [JAX](https://docs.jax.dev). The same experiment modules, kinetic
models and parameter constraints then run as *differentiable, compilable*
functions. Nothing about normal ChemEx changes: `chemex fit`, TOML files,
models, optimizers and outputs are untouched, and the NumPy results are
byte-identical to upstream ChemEx.

## When to use it

Use **plain ChemEx** (`chemex fit`, `chemex simulate`) for standard analyses:
fitting, grid searches, Monte Carlo/bootstrap, MCMC, plots and reports. The
JAX backend does not replace any of that, and the CLI does not use it.

Use the **JAX backend** from Python when you need one of the following:

| You want to… | Why JAX helps |
| --- | --- |
| Get **exact Jacobians** or gradients of profiles, residuals or χ² | Automatic differentiation gives derivatives accurate to ~1e-12 or better. Finite differences of NumPy ChemEx are limited by rounding noise: for weakly determined parameters (cross-relaxation rates, tight binding) they can be off by 1e-5 to 1e-3. |
| Study **identifiability** or compute the **Fisher information** `JᵀWJ` | One Jacobian call gives the information matrix, its eigenvalues (poorly identified parameter combinations), and linearised standard errors. At ChemEx's own fitted values these match ChemEx's reported uncertainties. |
| Do **experiment design** (which delays, offsets, fields help most) | Exact Jacobians at candidate measurement points predict the precision of any planned design before you record it ([recipe](#designing-experiments)). |
| Run **gradient-based samplers** (HMC/NUTS) or custom optimizers | `jax.grad` of a log-density built from ChemEx's own χ² works directly with JAX-based samplers. |
| **Simulate many parameter sets** (training data for ML, prior predictive checks, scans) | `jax.vmap` evaluates thousands of parameter vectors in one compiled call. |
| Write **your own objective** (penalties, priors, global constraints) | The weighted residuals are an ordinary JAX function you can combine freely. |

**Do not use it** when a single, plain fit is all you need: compilation costs
seconds to minutes (see [Costs and limits](#costs-and-limits)), which plain
ChemEx does not pay.

## Advantages over plain ChemEx

* **Exact derivatives instead of finite differences.** Every experiment type
  and every kinetic model (Eyring, binding, oligomerization, H/D, N-state,
  `.mf`/`.rs`/`.tc`) is differentiable. ChemEx's helper functions have JAX
  counterparts that use the same formulas. The oligomerization root solve is
  differentiated with the implicit function theorem.
* **The same science, verified.** The JAX profiles reproduce ChemEx NumPy to
  ≤ 1e-9 relative (typically 1e-16 – 1e-13) for all 37 experiment types. The
  weighted residuals and χ² reproduce ChemEx's native fit residuals (same
  scaling, masks, error estimates and ordering) on every shipped example.
  Derivatives were checked against 30–90-digit reference calculations.
* **Fast repeated evaluation.** After compilation, a full example's residuals
  are 2–6× faster than ChemEx's native evaluator. A batch of profiles via
  `vmap` costs about 0.2 ms per profile, against 0.6 ms for a NumPy profile.
* **Composable.** `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd` and
  `jax.jacrev` work on everything, and can be nested.

Measured on one CPU (Intel i7-13700H, JAX 0.11.2):

| Operation | Time | First call (compile) |
| --- | --- | --- |
| ChemEx NumPy, one 26-point CPMG profile | 0.6 ms | — |
| JAX, same profile | 1.1–1.9 ms | 2 s |
| JAX, profile + exact 6-parameter Jacobian | 7 ms | 8 s |
| JAX, `vmap` batch of 256, per profile | 0.2 ms | 2 s |
| ChemEx native residuals, CEST_15N (32 profiles) | 107 ms | — |
| JAX residuals, same | 18 ms | 3.5 s |

## Installation

The backend needs JAX ≥ 0.11 (CPU builds are enough):

```shell
pip install "jax>=0.11"
```

In a development checkout, you can instead run commands through
`uv run --with "jax>=0.11" …`. ChemEx itself, including `import chemex` and
the CLI, works without JAX.

## Check it on your own data

Before relying on the JAX backend for a data set, run the comparison command
on it. It takes the same inputs as `chemex fit`:

```shell
chemex compare-backends -e Experiments/*.toml -p Parameters/parameters.toml \
    [-d 2st] [--include ...] [--exclude ...] [--gradients 3] [--json report.json]
```

It builds your experiments exactly as `chemex fit` does, then evaluates them
twice: once with ChemEx's native NumPy evaluator, once with the JAX backend.
Each experiment is compiled separately, to keep memory low. For each
experiment it reports how closely the two agree, and the timings:

```text
                          NumPy vs JAX backend
Experiment  Prof.  Kern.  Calc.  Resid.  χ²     NumPy  JAX    Compile  OK
──────────────────────────────────────────────────────────────────────────
500mhz      54     1      6e-14  4e-14   6e-15  32 ms  23 ms  4.1 s    yes
800mhz      54     1      1e-13  5e-14   1e-14  27 ms  12 ms  3.2 s    yes
```

* **Calc.**: largest relative difference of the calculated intensities over
  all profiles.
* **Resid.**: difference of the weighted residuals that `chemex fit`
  minimises, relative to the weighted signal.
* **χ²**: relative difference of χ².
* **Kern.**: number of compiled kernels (profiles that share an experiment
  layout share one).

With `--gradients N`, it also compares JAX derivatives with finite differences
of the NumPy residuals, for the N fitted parameters shared by the most
profiles. Parameters with a negligible effect are marked "FD noise-limited":
there, finite differences cannot certify anything. The command exits with
status 1 if any parity check exceeds `--rtol` (default 1e-9), so it can also
run in scripts.

## Quick start

Build experiments with ChemEx's normal setup code (the same TOML files you
pass to `chemex fit`), then compile them:

```python
import chemex.jax as cj  # import first: switches JAX to float64
import jax
import jax.numpy as jnp
from pathlib import Path

from chemex.configuration.methods import Method, Selection
from chemex.configuration.parameters import read_defaults
from chemex.experiments.builder import build_experiments
from chemex.parameters.parameterization import ParameterRole
from chemex.parameters.spin_system import SpinSystem
from chemex.runtime import AnalysisSession, ensure_plugins_registered

example = Path("examples/Experiments/CPMG_15N_IP")
ensure_plugins_registered()
session = AnalysisSession()
session.set_model("2st")  # as `chemex fit -d 2st`
experiments = build_experiments(
    sorted((example / "Experiments").glob("*.toml")),
    Selection(include=None, exclude=None),
    session=session,
)
# Optional: restrict to some residues, like a method step's INCLUDE.
experiments.select(
    Selection(include=[SpinSystem.from_name(r) for r in ("15", "31")], exclude=None)
)
session.parameters.set_defaults(read_defaults([example / "Parameters/parameters.toml"]))
session.try_build_analysis_values()
parameterization = session.compile_parameterization(Method(), experiments.param_ids)
frame = parameterization.frame_from_snapshot(session.analysis_values.snapshot())

# The parameters to vary (here: those ChemEx would fit), as parameter IDs.
free_ids = [
    i
    for i in parameterization.independent_ids
    if parameterization.role(i) == ParameterRole.FIT
]
residuals = cj.compile_residuals(experiments, parameterization, free_ids, frame=frame)

x0 = residuals.x0  # current values of free_ids
r = jax.jit(residuals)(x0)  # ChemEx's weighted residuals
chi2 = jax.jit(residuals.chi2)(x0)  # Σ r²
J = jax.jit(jax.jacfwd(residuals))(x0)
```

Build all residues first and select afterwards, as above, rather than
selecting at build time: with `error = "duplicates"` or `"scatter"`, ChemEx
estimates the noise from all profiles of an experiment, and the residuals
then carry exactly the errors `chemex fit` uses.

`x` is always a float64 vector of the values of `free_ids`, in that order.
All other independent parameters stay at their values in `frame`.

## Recipes

### Fisher information and standard errors

```python
F = J.T @ J  # W = diag(1/σ²) is already in the residuals
eigenvalues = jnp.linalg.eigvalsh(F)
standard_errors = jnp.sqrt(jnp.diag(jnp.linalg.inv(F)))
```

A complete script, which also prints the least-determined parameter
combination, is in
[`examples/jax/fisher_cpmg_15n_ip.py`](https://github.com/debadutta-patra/ChemEx-JAX/tree/jax-backend/examples/jax).
Evaluated at ChemEx's fitted STEP1 values it reproduces ChemEx's χ² (434.56)
and its reported uncertainties, for example KEX_AB ±6.234 and PB ±8.024e-4.

### Designing experiments

Because derivatives are exact and cheap, you can predict how precisely a
planned experiment will determine the parameters *before* recording it. Then
you can choose the measurements that maximise that precision. This is local
optimal design: it assumes a prior guess of the parameters and a noise level.

The key step is to compute ChemEx's profile for **candidate** measurement
points: every CPMG `ncyc`, CEST offset or relaxation delay you could record.
Copy an existing profile and give it the candidate points as metadata; no data
are needed. One exact Jacobian then gives the information contributed by
every candidate. The Fisher information of any design (which points, how
many repeats) is a weighted sum of rows, so comparing designs requires no
further compilation:

```python
from copy import deepcopy
import numpy as np
from chemex.containers.data import Data


def with_points(profile, metadata, **settings):
    """Copy of `profile` that computes `metadata` (optionally with new settings)."""
    candidate = deepcopy(profile)
    n = metadata.size
    candidate.data = Data(exp=np.ones(n), err=np.ones(n), metadata=metadata)
    if settings:
        sequence = candidate.pulse_sequence
        sequence.settings = sequence.settings.model_copy(update=settings)
    return candidate


ncyc = np.arange(0.0, 31.0)  # reference + every ncyc up to 1 kHz at T2 = 30 ms
free = ["__KEX_AB", "__PB", "__DW_AB_15N", "__R2_A_15N_500_0MHZ"]
f = cj.compile_profile(with_points(profile, ncyc), parameterization, free, frame=frame)

theta = f.x0  # prior guess (from your parameter file)
intensities = np.asarray(f(theta))
# Rows: one per candidate; last column = the intensity scale ChemEx always fits.
rows = np.column_stack([np.asarray(jax.jacfwd(f)(theta)), intensities])
sigma = 0.008 * intensities[0]  # assumed noise: 0.8 % of the reference


def predicted_se(counts):  # counts[k] = repeats of candidate k
    fisher = (rows.T * (counts / sigma**2)) @ rows
    return np.sqrt(np.diag(np.linalg.inv(fisher)))
```

`predicted_se(counts)[:2]` gives the predicted standard errors of KEX_AB and
PB for any choice of points. A greedy loop (add the measurement that most
reduces the variances you care about, repeat) finds a good design in
milliseconds. Settings that change the compiled program, such as the CPMG
period, the CEST B1 field or the spectrometer field, are compared by
compiling one candidate per setting (`with_points(profile, ncyc,
time_t2=0.04)`), which costs a few seconds each.

The complete script
[`examples/jax/design_cpmg_15n_ip.py`](https://github.com/debadutta-patra/ChemEx-JAX/tree/jax-backend/examples/jax)
does this for one residue of the CPMG_15N_IP example at 500 MHz. It assumes
the example's own parameter values and noise (0.8%), and uses 26 points:

| Design (26 points) | Predicted SE(KEX) | Predicted SE(PB) |
| --- | --- | --- |
| The example's ncyc list | 23.8 % | 9.6 % |
| Greedy optimum (ncyc 0×2, 1×6, 2×8, 5×4, 6×2, 8, 16, 23, 30) | 15.5 % | 4.9 % |
| Greedy optimum with T2 = 60 ms instead of 30 ms | 7.0 % | 2.3 % |

#### Choosing the next experiment from data you already have

Usually you already have some data, say CPMG at one B0 or CEST at one B1, and
need to decide what to record next. Fisher information adds over independent
experiments, so the predicted precision after the next experiment is

```text
F_next = F_existing + F_candidate
```

* `F_existing` is `JᵀJ` of ChemEx's weighted residuals of the data you have,
  at your current estimates (use your fitted values). It uses the real error
  bars, and the intensity scale is profiled exactly as in the fit.
* For each candidate setting (another B0, another B1…), copy your experiment
  TOML with that one setting changed and build it **together** with the
  existing data. New field-dependent parameters, such as R2 at a new B0, are
  then created and accounted for. Fill the candidate's data with ChemEx's own
  noiseless prediction at the current estimates, with errors equal to your
  current relative noise. `F_candidate` is then `JᵀJ` of the candidate's
  weighted residuals.

The core of it, once the existing and candidate experiments are built jointly
(`joint`, `parameterization`, `frame`) and split into `existing` and
`candidate` containers:

```python
from chemex.parameters.program_evaluation import independent_dependencies


def fisher(experiments, free):
    r = cj.compile_residuals(experiments, parameterization, free, frame=frame)
    J = jax.jit(jax.jacfwd(r))(r.x0)
    return np.asarray(J.T @ J)


# Candidate data = ChemEx's noiseless prediction, with the current relative noise.
values = parameterization.resolve(frame)
for experiment in candidate:
    for profile in experiment.profiles:
        calc = np.asarray(profile.calculate_unscaled(values))
        profile.data.exp = calc / np.max(np.abs(calc))
        profile.data.err = np.full_like(calc, relative_noise)
        profile.data.mark_dirty()

free = [
    i
    for i in dict.fromkeys(
        i
        for e in joint
        for p in e.profiles
        for i in independent_dependencies(parameterization, p.name_map.values())
    )
    if parameterization.role(i) == ParameterRole.FIT
]
F = fisher(existing, free) + fisher(candidate, free)
se = np.sqrt(np.diag(np.linalg.inv(F)))  # predicted standard errors
```

Repeat for every candidate setting and pick the one that most reduces the
errors you care about. The script
[`examples/jax/next_experiment.py`](https://github.com/debadutta-patra/ChemEx-JAX/tree/jax-backend/examples/jax)
does all of it: writing the candidate TOML and placeholder data, the joint
build, initialising the new-field R2 values from the existing field, and the
ranking. It has two modes:

```shell
# CPMG: which B0 to add to a 15N CPMG data set recorded at 500 MHz?
uv run --with "jax>=0.11" python examples/jax/next_experiment.py cpmg-b0 \
    --parameters my/parameters.toml my/Output/STEP1/Parameters/fitted.toml
# CEST: which B1 to add to a 15N CEST data set recorded with B1 = 26 Hz?
uv run --with "jax>=0.11" python examples/jax/next_experiment.py cest-b1 \
    --values 5 10 13 20 40
```

With the examples' own parameter values, the predicted relative standard
errors of KEX_AB and PB after adding each candidate are:

| CPMG_15N_IP (5 residues), next B0 | SE(KEX_AB) | SE(PB) |
| --- | --- | --- |
| none (500 MHz only) | 6.2 % | 2.5 % |
| 600 MHz | 2.5 % | 0.9 % |
| 800 MHz | 1.4 % | 0.8 % |
| 1200 MHz | 1.1 % | 0.8 % |

| CEST_15N (3 residues), next B1 | SE(KEX_AB) | SE(PB) |
| --- | --- | --- |
| none (26 Hz only) | 29.6 % | 21.5 % |
| 5 Hz | 9.0 % | 6.3 % |
| 10 Hz | 6.3 % | 4.0 % |
| 13 Hz | 7.0 % | 4.4 % |
| 20 Hz | 12.9 % | 8.6 % |
| 40 Hz | 12.6 % | 10.2 % |

So a second B0 helps most when it is well separated from the first, with
diminishing returns above about 800 MHz here. For CEST, a weaker second B1
(10–13 Hz) complements 26 Hz far better than a stronger one. As a sanity
check of the method, "repeating" the 500 MHz CPMG experiment lowers the
errors by close to √2 (6.2 → 4.5 %, 2.5 → 1.9 %), as it should.

The noise assumption matters here. By default each candidate has the same
*relative* noise as your existing data. Sensitivity usually grows with B0, so
for field comparisons pass `--snr-exponent 1.5` (relative noise ∝ B0^-1.5)
or whatever matches your probes and sample. Equally, the measurement time
each option needs (more B1 fields, longer relaxation delays) is a cost the
prediction does not include.

Treat such results as guidance, not prescriptions:

* **They are only as good as the assumptions.** The prediction holds for the
  assumed parameters, model and noise. Check robustness by scoring a design
  over draws from your prior: `jax.vmap(jax.jacfwd(f))` computes the Jacobian
  for all draws in one call. In the example the optimum stays better over
  ±30% prior uncertainty in KEX and PB (median 16.3% against 23.5%).
* **Optimal designs concentrate points.** The greedy optimum above repeats a
  few ncyc values. That is efficient if the two-state model is right, but
  leaves little data to detect a wrong model. Keep some spread, and model
  checks in mind.
* **Include everything the fit will estimate.** Leaving a parameter (or the
  intensity scale) out of `rows` makes the prediction optimistic.
* **Settings that change relaxation losses matter.** Longer CPMG periods give
  more dispersion but less signal; the trade-off depends on R2 and is exactly
  what the prediction captures.

### Your own optimizer with exact Jacobians

```python
import numpy as np
from scipy.optimize import least_squares

fun = jax.jit(residuals)
jac = jax.jit(jax.jacfwd(residuals))
fit = least_squares(
    lambda x: np.asarray(fun(x)), x0, jac=lambda x: np.asarray(jac(x)), method="lm"
)
residuals.validate(fit.x)  # ChemEx's domain checks on the result
```

### Many simulations at once

```python
from chemex.parameters.program_evaluation import independent_dependencies

profile = next(p for e in experiments for p in e.profiles)
ids = independent_dependencies(parameterization, profile.name_map.values())
simulate = cj.compile_profile(profile, parameterization, ids, frame=frame)

key = jax.random.PRNGKey(0)
batch = simulate.x0 * (1 + 0.05 * jax.random.normal(key, (1000, len(ids))))
profiles = jax.jit(jax.vmap(simulate))(batch)  # shape (1000, n_points)
```

`compile_profile` returns the *unscaled* profile, as ChemEx's
`calculate_unscaled`.

### Gradient-based sampling (HMC/NUTS)

```python
def log_density(x):
    return -0.5 * residuals.chi2(x)  # add log-priors and bounds as needed


grad = jax.jit(jax.grad(log_density))
```

`log_density` can be passed to any JAX-based sampler. Samplers are not
installed with ChemEx; choose one yourself.

## Reference

| Object | Description |
| --- | --- |
| `cj.compile_residuals(experiments, parameterization, free_ids, *, frame)` | Returns `r(x)`: ChemEx's native weighted residual vector. Attributes: `x0`, `free_ids`, `size`, `group_count`; methods `chi2(x)`, `validate(x)`. |
| `cj.compile_profile(profile, parameterization, free_ids, *, frame)` | Returns `f(x)`: one profile's unscaled calculated intensities. Attributes: `x0`, `free_ids`; method `validate(x)`. |
| `validate(x)` | Runs ChemEx's own checked parameter resolution on a concrete `x` and raises on domain errors (e.g. populations summing to more than 1). |
| `cj.jacobian(f, x, chunk_size=None, mode="forward")` | Jacobian of `f` at `x`. Without `chunk_size` it is `jax.jacfwd`/`jacrev`; with it, columns (forward) or rows (reverse) are computed `chunk_size` at a time to bound memory. |
| `residuals.calculations(x)` | Unscaled calculated intensities of every observation, in ChemEx's order. |
| `cj.release_memory()` | Drops compiled kernels and returns freed memory to the operating system. |
| `cj.kernel_signature(profile)`, `cj.KERNEL_CACHE` | Grouping key and cache of compiled kernels (advanced). |

`free_ids` must be **independent** parameter IDs of the parameterization
(`parameterization.independent_ids`), for example `__KEX_AB` or `__PB`.
Derived parameters follow from them through the model's constraints.

## Costs and limits

**Compilation time.** The first call of a compiled function for a given
experiment layout takes seconds (CPMG, CEST: 2–15 s). Long shaped-pulse
sequences take up to minutes (COSCEST: ~2 min). Profiles that share a layout
(same experiment settings and data points; only the residue differs) are
compiled once and evaluated together. For example, 108 CPMG profiles at two
fields compile as 2 kernels.

**Memory.** Compilation and especially forward-mode Jacobians are
memory-hungry: a Jacobian carries one extra copy of the calculation per
parameter. Measured peaks:

* one profile with a 6-parameter Jacobian: under 1 GB;
* a whole example's residuals: 1–5 GB; the largest example, 468 D-CEST
  profiles, needs about 8 GB;
* a Jacobian over 20–25 parameters: 2–7 GB;
* a Jacobian over *every* parameter of a large CEST example: over 30 GB.
  **Avoid this.**

Keep Jacobians to fitting-step size (tens of parameters), or compute them in
chunks with `cj.jacobian(r, x, chunk_size=16)`: at most 16 columns are held at
once. For CPMG_15N_IP with all 326 parameters free, this cuts peak memory from
5.9 GB to 1.7 GB for about 16% more time. For few outputs and many parameters,
use `mode="reverse"`. For a scalar such
as χ² or a log-density, prefer `jax.grad`/`jacrev`. In long-running processes
call `cj.release_memory()` when idle. Otherwise the process keeps its peak
memory after the compiled functions are dropped.

**Double precision only.** Importing `chemex.jax` switches JAX to float64,
and float32 inputs are rejected. Single precision gives falsely small error
bars in fast exchange. Import `chemex.jax` before creating any JAX arrays.

**Hardware.** Experiments that use complete B1 dephasing (`b1_distribution =
{ type = "dephasing" }`, most CEST setups) need an eigendecomposition. JAX
provides it on CPU, and the fork has only been tested on CPU. Other
experiments use a matrix exponential that also runs on GPUs.

**Domain checks happen outside the compiled function.** Inside `jit`/`grad`,
values are abstract, so ChemEx's checks (finite values, populations within
[0, 1], positive concentrations…) cannot run there. Call `validate(x)` on
concrete parameter vectors that matter: starting points, optimizer results,
accepted samples.

## What can and cannot be differentiated

* **Differentiable:** every independent parameter of every kinetic model and
  experiment: chemical shifts, rates, populations, exchange rates,
  thermodynamic and binding constants, model-free parameters.
* **Not differentiable (fixed constants):** experiment settings and
  conditions, such as delays, pulse widths, B1 fields, carriers, offsets,
  temperature, protein and ligand concentrations and field strength. They
  determine the compiled program.
* **At boundaries:** where an exchange pathway is switched off exactly
  (a rate or equilibrium constant equal to 0) or a population is exactly 0,
  the model is not differentiable. There JAX returns the derivative of the
  branch ChemEx selects (for example zero with respect to the switched-off
  pathway).
