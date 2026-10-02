# JAX backend port — working notes

> **Modified fork.** This repository is a GPL-3.0-or-later fork of ChemEx by
> Guillaume Bouvignies (<https://github.com/gbouvignies/ChemEx>). These notes
> describe the fork-only work that makes the simulation path backend-agnostic
> (NumPy default, optional JAX).

## 1. Baseline

| Item | Value |
| --- | --- |
| Base commit | `d0ba6e34` — "Publish complete runtime-checked kinetic-model reference (#794)" (upstream `main`, 2026-09-26; includes #793, complete B1 dephasing as zero modal weights) |
| Remotes | `origin` = fork (`debadutta-patra/ChemEx-JAX`), `upstream` = `gbouvignies/ChemEx` |
| Feature branch | `jax-backend` |
| Python | 3.13.12 (conda-forge build), managed by `uv sync --locked` |
| NumPy / SciPy | 2.5.1 / 1.18.0 (from `uv.lock`) |
| JAX (Phase 0 only) | jax/jaxlib 0.11.2 (CPU; an NVIDIA GPU is present but no CUDA jaxlib is installed), as an ephemeral `uv run --with "jax>=0.11"` overlay; **not** added to `pyproject.toml`/`uv.lock` until Phase 5 |
| Hardware | Intel i7-13700H (20 threads), Linux 7.0, CPU only |

## 2. Golden outputs (NumPy byte-identity oracle)

`tests/backend/golden_outputs.py` runs every `examples/*/*/run.sh` *unmodified*
through a `chemex` shim that redirects `-o`, and forces
`--plot nothing --workers 1 --native-threads 1` plus
`OMP/OPENBLAS/MKL_NUM_THREADS=1`. Every output file except `run_info/` is
hashed into `tests/backend/golden_manifest.json` (committed). The outputs
themselves live in `.golden/` (git-ignored).

```sh
uv run python tests/backend/golden_outputs.py generate   # only on the base commit
uv run python tests/backend/golden_outputs.py compare    # after every change
```

Status on `d0ba6e34`: **34 examples** (30 `Experiments/*`, 4
`Combinations/*`), **452 hashed files**; generation takes ~21 min with 10
parallel jobs.

**Determinism finding.** A second full run of the *unmodified* base commit
differed from the first in 106 files, all `Statistics/**/evidence.json`. Every
differing value is under a key ending in `identity` (17 key names, e.g.
`occurrence_identity`, `accepted_result_identity`, `bundle_identity`): upstream
mints per-run `uuid4()` occurrence identities
(`parameters/parameterization.py:1690`, `optimize/uncertainty.py:7073-7082`,
`optimize/de_direct_trf.py:1094,1190`) and chains hashes from them. All
numeric content was identical. The tool therefore canonicalises `*.json`
files by blanking values of keys ending in `identity` before hashing; every
other JSON value and every non-JSON file (`Parameters/*.toml`, `Data/*`,
`statistics.toml`, …) is compared byte for byte. With that rule, two
independent runs of the base commit agree on all 452 files (0 differences).
`.golden/` and `.golden-check/` contain no timestamps or absolute paths
outside `run_info/`.

**Decision (maintainer, 2026-10-02):** all `*identity` values are excluded
from golden comparisons, including the six that happen to be stable across
runs (`policy_identity`, `accepted_evaluation_identity`,
`constraint_program_identity`, `calibration_identity`,
`evaluation_plan_identity`, `evaluator_parameterization_identity`).
Identities are provenance (per-run `uuid4` occurrence ids and hashes chained
from them, or fingerprints that also hash scientific-function *source files*
via `parameterization.py::_source_record`), not scientific results. Numerical
byte-identity is judged on every other value and file.

## 3. Upstream test baseline

`uv run pytest -q -n 8` on `d0ba6e34`: **2310 passed** in 636 s (both the
ordinary and `scientific_acceptance` layers), 0 failures. The only error was
collection of the untracked `prototypes/jax_bridge/test_bridge.py` (it imports
`jax`); a fork-only root `conftest.py` now sets
`collect_ignore = ["prototypes"]`.

## 4. Prototype reproduction

`prototypes/jax_bridge/test_bridge.py` hard-codes
`ROOT = /tmp/ChemEx/examples/Experiments`; it was run from a scratch copy with
`ROOT` pointing at this checkout (prototype files themselves unchanged).

The shipped `validation_results.txt` reports **26/30**, not 30/30: forward
parity holds everywhere, but four examples fail the prototype's 1e-5
gradient criterion. The reproduction (JAX 0.11.2, CPU) matches that exactly:

| Example | Model | Status | Fwd rel. err | Grad rel. err (shipped → reproduced) |
| --- | --- | --- | --- | --- |
| CEST_13C | 2st | OK | 3.3e-12 | 7.0e-09 → 7.0e-09 |
| CEST_13C_LABEL_CN | 2st | OK | 8.2e-15 | 8.7e-09 → 6.7e-09 |
| CEST_15N | 2st | OK | 9.6e-15 | 7.7e-10 → 5.5e-10 |
| CEST_15N_CW | 2st.mf | OK | 3.8e-13 | 1.3e-08 → 1.1e-08 |
| CEST_15N_LABEL_CN | 2st | OK | 6.5e-15 | 8.7e-11 → 6.6e-11 |
| CEST_15N_TR | 2st.mf | OK | 5.9e-14 | 2.2e-09 → 2.1e-09 |
| CEST_1HN_AP | 2st | OK | 6.9e-14 | 1.3e-07 → 1.6e-07 |
| CEST_1HN_IP_AP | 2st.rs | **MISMATCH** | 9.6e-12 | 1.2e-04 → 5.6e-05 |
| CEST_CH3_1H_IP_AP | 2st.rs | OK | 3.9e-11 | 4.9e-07 → 2.9e-07 |
| COSCEST_1HN_IP_AP | 2st.rs | OK | 8.4e-13 | 2.6e-06 → 3.3e-06 |
| CPMG_13CO_AP | 2st | **MISMATCH** | 1.6e-14 | 2.0e-05 → 1.9e-05 |
| CPMG_13C_IP | 2st | OK | 9.8e-10 | 1.2e-06 → 1.2e-06 |
| CPMG_15N_IP | 2st | OK | 1.1e-12 | 2.9e-08 → 2.9e-08 |
| CPMG_15N_IP_0013 | 2st | OK | 1.7e-10 | 4.1e-06 → 4.2e-06 |
| CPMG_15N_TR | 2st | OK | 1.2e-13 | 1.0e-08 → 7.0e-09 |
| CPMG_15N_TR_0013 | 2st | OK | 3.4e-10 | 3.5e-07 → 3.5e-07 |
| CPMG_1HN_AP | 2st | OK | 8.0e-14 | 2.3e-08 → 3.6e-08 |
| CPMG_1HN_AP_0013 | 2st | OK | 4.1e-14 | 5.9e-08 → 8.9e-08 |
| CPMG_CH3_13C_H2C | 2st | **MISMATCH** | 9.5e-14 | 2.4e-05 → 1.9e-05 |
| CPMG_CH3_13C_H2C_0013 | 2st | OK | 6.7e-10 | 5.1e-06 → 1.6e-06 |
| CPMG_CH3_1H_SQ | 2st | OK | 2.8e-13 | 7.8e-07 → 4.3e-07 |
| CPMG_CH3_1H_TQ | 2st | OK | 9.2e-14 | 8.9e-08 → 8.8e-08 |
| CPMG_CH3_1H_TQ_DIFF | 2st | OK | 7.7e-11 | 2.4e-07 → 2.4e-07 |
| CPMG_CH3_MQ | 2st | OK | 3.8e-14 | 2.6e-10 → 2.6e-10 |
| CPMG_CHD2_1H_AP | 2st | OK | 6.8e-14 | 8.8e-08 → 5.3e-08 |
| CPMG_HN_DQ_ZQ | 2st | **MISMATCH** | 8.2e-10 | 2.9e-05 → 3.0e-05 |
| DCEST_15N | 2st | OK | 6.7e-12 | 4.4e-07 → 4.4e-07 |
| DCEST_15N_HD_EXCH | 2st_hd | OK | 1.2e-11 | 5.0e-07 → 5.0e-07 |
| RELAXATION_HZNZ | 2st | OK | 1.4e-15 | 4.6e-09 → 4.6e-09 |
| RELAXATION_NZ | 2st | OK | 1.2e-16 | 2.4e-13 → 2.4e-13 |

Forward errors match the shipped file to the printed digit except where the
reference FD is noisy (gradient column). The prototype's gradient check is
only a single relative number per profile; it does not separate "weak
parameter" from "real error". Phase 2 will diagnose the four mismatches with
the brief's criterion (effect ≥ 1e-6 ⇒ ≤ 1e-5, plus a JAX-FD cross-check and
`jacfwd`/`jacrev` agreement).

Timing, CPMG_15N_IP, one 26-point profile, 6 free parameters (idle machine,
`timing.py` built on the prototype):

| Operation | Brief | Reproduced |
| --- | --- | --- |
| ChemEx forward | 1.1 ms | 0.99 ms |
| JAX jit forward | 1.2 ms | 1.00 ms (compile 2.0 s) |
| JAX forward + full `jacfwd` | 3.3 ms | 3.05 ms (compile 7.4 s) |
| `vmap` batch of 256, per profile | 0.27 ms | 0.32 ms (compile 2.9 s) |
| First compile per layout | 3–8 s | 2.0–7.4 s |

## 5. Inventory of NumPy-only constructs on the evaluation path

Evaluation path: `Profile.calculate_unscaled` → `Spectrometer.update` →
`ISLiouvillianEngine.update/_build_base_liouvillian` → `l_free` →
`PulseKernel` / `PulseLibrary` → `calculate_propagators` → sequence
`calculate` → `detect`. Categories:

- **C** concretisation (`float()`, `.item()`, `int()`, `math.*`, Python `if` on a traced value)
- **I** in-place update of a traced value (`+=`, `a[idx] = …`)
- **L** list indexing (`a[[i, j]]`), rejected by JAX
- **N** NumPy call on traced arrays (`np.array([...])`, `np.matmul`, `np.mean`, `np.stack`, `numpy.linalg.matrix_power`, `scipy.linalg.expm`, `np.linalg.eig`)
- **K** cache that could hold traced values
- **D** non-differentiable algorithm (eigenvectors of a non-symmetric matrix)
- **OK** data-derived constant: stays NumPy/Python by design

### 5.1 NMR engine (`src/chemex/nmr/`)

| Site | Construct | Cat. | Plan |
| --- | --- | --- | --- |
| `_pulses/propagators.py:40-72` `calculate_propagators` | `np.asarray(dtype=float64)`, `scipy.linalg.expm`, `np.linalg.eig` + `solve`, `np.exp(..., where=)` | N, D | `backend.propagators`; NumPy backend calls this function verbatim; JAX uses `expm` (Padé) and a `custom_jvp` dephased propagator |
| `_pulses/propagators.py:75-107` `make_perfect180/90`, `get_phases` | `cachetools.cached` keyed on `engine.basis` | K (constants) | OK: depends on basis only; stays NumPy |
| `_pulses/kernel.py:33` `add_phases` | `np.array([phases[i] @ p @ phases[-i] ...])` | N | `backend.stack` |
| `_pulses/kernel.py:47` `_phase_mixed_term` | `np.cos/np.sin(phase)` | OK | phase is a constant |
| `_pulses/kernel.py:107-110` `shaped_pulse_i` | `set(pairs)` (constants, OK), `reduce(np.matmul, …)` | N | `reduce(operator.matmul, …)` (bit-identical) |
| `_pulses/library.py:88,92,96,100` | `p90_i[[3,0,1,2]]`, `p90_i[[1,2,3,0]]` | L | module-level `np.array` index constants |
| `_pulses/library.py:64-69,102-112` | per-generation pulse caches (`_ISpinPulseCache`, `_SSpinPulseCache`) | K | hold traced values on a JAX workspace → JAX evaluation always uses a *private per-trace copy* of the spectrometer (see §6.2) |
| `_pulses/library.py:122-144` `p9024090_nh` | `np.diff/np.sort` on pulse widths; `if pw… <= pw…` | OK | pulse widths are settings |
| `_engine/engine.py:97-104` `_build_base_liouvillian` | `sum(matrix * par_value, start=np.zeros)` | N (works via `__array_priority__`, but implicit) | explicit `xp` via backend |
| `_engine/engine.py:127-186` `ppm_*`, `carrier_*`, `offset_*` setters | `float(value)`, `np.sign` | C | OK: values are settings / metadata (data-derived), never fitted |
| `_engine/engine.py:271-274` `gradient_dephasing` setter | `float(value)` | C | OK: setting (`k2_factor`) |
| `_engine/engine.py:277-288` `l_free` | `sum(..., start=np.zeros)` | N (implicit) | explicit `xp` |
| `_engine/engine.py:291` `weights` | products of distribution weights | OK | constants |
| `_engine/magnetization.py:46-51` `build_equilibrium_magnetization` | `magnetization += vec * p_state` | I | out-of-place `m = m + …` (same ufunc, same operand order ⇒ bit-identical) with `xp.zeros` start |
| `_engine/magnetization.py:61-75` `build_start_magnetization` | `magnetization += sign * pop * ratio * vector` | I | as above |
| `_engine/magnetization.py:34-38` `detect_signal` | `np.iscomplexobj`, `float(detected.item())` | C | `backend.detect`; NumPy keeps `float(...)`; JAX returns a 0-d array |
| `_engine/magnetization.py:84-89` `keep_components` | `keep_mask[keep_mask > 0] = 1.0` | I | OK: mask is a basis constant; only `mask * magnetization` touches traced data |
| `_engine/readout.py:30` `detect` | delegates to `detect_signal` | — | route through backend |
| `_engine/effective_field.py:33-34` | `float(np.arctan2(w1, wi))` with `wi` depending on fitted `cs_i_*` | C | `xp.arctan2`, no `float` on the JAX path (used only by `wip/relaxation_15n_r1rho`) |
| `_engine/effective_field.py:77-82` | `np.array([[cos, -sin], [sin, cos]])` of traced angle; `m[..., [ix, iz], :] = …` | N, L, I | `xp.stack`, functional `.at[].set` on JAX |
| `_engine/analysis.py:25` `calculate_shifts` | `np.linalg.eigvals(...).imag` | N | `xp.linalg.eigvals` (JAX JVP of eigen*values* is supported) — used by `shift_*` |
| `_engine/analysis.py:33-44` `calculate_r1rho` | `eigvals`, boolean-mask selection by value, `float(np.max)`, `if size == 0: raise` | N, C | masked `xp.where` + `max`; the "no real eigenvalue" error becomes a concrete post-check (wip only) |
| `_engine/views.py:14` | `np.squeeze` + shape check | OK on shape | use `.squeeze()`/`.reshape` method form |
| `spectrometer.py:143-150` `identity`, `zfilter` | constants | OK | — |
| `nmr/rates.py:24-38, 74-163, 186-262` | arithmetic on `tauc`, `s2`, `khh`; `np.array([... wi ...])` uses `h_frq` (an expression **literal**) only | OK | already namespace-generic as long as `h_frq` stays a Python literal; `rates["r2_s"] += khh` rebinds a dict entry (not an array in-place op) |

### 5.2 Experiment catalog (`src/chemex/experiments/catalog/`)

Every module ends with `return np.array([... spectrometer.detect(...) ...])`
(category **N**): 35 + 3 WIP sites, e.g. `cpmg_15n_ip.py:139`,
`cest_15n.py:114`, `relaxation_nz.py:99`, `noesyfpgpph19.py:80`,
`shift_15n_sq.py:74`. Plan: `spectrometer.backend.stack(...)` (NumPy:
`np.array(seq)`, identical).

`from numpy.linalg import matrix_power` (**N**), 13 modules:
`cest_1hn_ip_ap.py:138`, `coscest_13c.py:127`, `coscest_1hn_ip_ap.py:129`,
`cpmg_13c_ip.py:131`, `cpmg_13co_ap.py:157`, `cpmg_15n_ip.py:133`,
`cpmg_15n_tr.py:153`, `cpmg_1hn_ap.py:141`, `cpmg_ch3_13c_h2c.py:137`,
`cpmg_ch3_mq.py:113`, `cpmg_chd2_1h_ap.py:129`, `dcest_13c.py:181`,
`dcest_15n.py:193`. Exponents are data/settings (`int(ncyc)`, `eta_block`,
`ncyc_dante`, `double_periods`). Plan: `spectrometer.backend.matrix_power`.

List indexing (**L**) — more sites than the original brief listed:

| Module | Lines |
| --- | --- |
| `cpmg_15n_ip_0013.py` | 189, 191 |
| `cpmg_15n_tr.py` | 143 (×2), 152 |
| `cpmg_15n_tr_0013.py` | 220 (×2), 230, 232 |
| `cpmg_1hn_ap_0013.py` | 214, **215, 219** (on `perfect90_i`/`perfect180_i`), 222 |
| `cpmg_ch3_13c_h2c.py` | 136 |
| `cpmg_ch3_13c_h2c_0013.py` | 244, 246, 252, 254, 273, 274 |
| **`cpmg_ch3_1h_tq_diff.py`** | 205 (×2) — not in the brief |
| **`cpmg_hn_dq_zq.py`** | 148, 149 — 2-D `a[[0, 1], [0, 1]]` — not in the brief |
| `nmr/_pulses/library.py` | 88, 92, 96, 100 |

(`cpmg_ch3_1h_sq.py:114-115` and `nmr/constants.py:261` build constant
arrays with `np.array([[...]])`; not indexing.)

Other **N** sites on traced arrays:

| Site | Construct | Plan |
| --- | --- | --- |
| `cpmg_15n_ip_0013.py:199`, `cpmg_15n_tr_0013.py:241-242`, `cpmg_1hn_ap_0013.py:230-231`, `cpmg_ch3_13c_h2c_0013.py:270-276`, `cpmg_ch3_1h_dq.py:175-176`, `cpmg_ch3_1h_sq.py:168-172`, `cpmg_ch3_1h_tq.py:155-156`, `cpmg_ch3_1h_tq_diff.py:225-226`, `cpmg_hn_dq_zq.py:160-161` | `reduce(np.matmul, echo[phases])` | `reduce(operator.matmul, …)` — bit-identical on NumPy, no `xp` needed |
| `cpmg_15n_tr.py:143`, `cpmg_15n_tr_0013.py:219` | `np.mean(props, axis=0)` | `props.mean(axis=0)` (NumPy's `np.mean` dispatches to the same method) |
| `cpmg_1hn_ap_0013.py:217` | `np.stack([...])` of propagators | `backend.stack` / `xp.stack` |
| `shift_13c_sq.py:57-60`, `shift_15n_sq.py:57-60`, `shift_15n_sqmq.py:56-59` | `_find_nearest`: `np.asarray`, `argmin` on eigenvalues derived from fitted shifts | `xp` version; the selected index is piecewise constant (OK for AD) |
| `wip/relaxation_15n_r1rho_eig.py:86` | `np.exp(-r1rho * times)` | `xp.exp` |

Data-derived constants that stay NumPy/Python (**OK**): `ncycs`/`offsets`
from `data.metadata`, `set(ncycs)`, `dict` keys from delays (`delays =
dict(zip(all_delays, spectrometer.delays(all_delays)))`), `float(ncyc)` cache
keys, `np.take/np.flip/np.arange/np.floor/np.unique` on phase indices,
`self._phase_cache` (phase-index arrays only), `is_reference`,
`coscest_*:122-125` (`np.cos` on the cosine-modulation grid),
`dcest_*.py:74-89` (`float(pw90)` etc., settings).

### 5.3 Parameter program and scientific functions

| Site | Construct | Cat. | Plan |
| --- | --- | --- | --- |
| `parameters/parameterization.py:847-884` `_resolve_values` | ordered evaluation over `_ordered_constraints` (private) | — | backend-generic evaluator walking the same `_ordered_constraints`, same tree semantics |
| `parameterization.py:1781-1802` `_finite_scalar` | `isinstance(value, Real)`, `float()`, `math.isfinite`; maps `-0.0 → 0.0` | C | concrete post-validation pass; traced path uses `x + 0.0` (maps `-0.0 → +0.0`, derivative 1) |
| `parameterization.py:1938` `_evaluate_expression` | `np.errstate(all="raise")` | C | concrete post-validation |
| `parameterization.py:1827-1858` `_evaluate_binary` | `try/except ArithmeticError` (division by zero) | C | post-validation |
| `parameterization.py:573` `functions["max"] = max` | builtin `max` | C | twin `jnp.maximum` (sub-gradient at ties: JAX gives 0.5/0.5) |
| `models/kinetic/_eyring.py` | `math.*`, `float()`, domain `if`s | C | JAX twins; checks → post-validation (Phase 3) |
| `models/kinetic/_binding.py` | `math.isfinite`, `math.fsum` with correction terms, `_logsumexp`, many value branches, `-inf` sentinels | C | twins (Phase 3); parity is tolerance-based (≤1e-10), not bitwise, because of `fsum` corrections |
| `models/kinetic/_oligomerization.py` | `scipy.optimize.brentq` | C | bracketed solve + implicit-function-theorem `custom_jvp` |
| `models/kinetic/settings_*.py` (binding, oligomerization, Eyring families) | `functools.lru_cache` on the `calculate_*` user functions | K | hashing a tracer fails → twins bypass the cached callables entirely |

### 5.4 Scientific functions (`ScientificFunctionBinder.for_model`)

`for_model` binds `rate_functions | user_function_registry.get(model)` plus
`max`. Function ids per model (enumerated at runtime with all plugins
registered; twin coverage is tracked here from Phase 3 on):

| function_id | Implementation | Models | Twin |
| --- | --- | --- | --- |
| `nh`, `nh_d`, `hn`, `hn_d`, `ch`, `ch_d`, `hc`, `hc_d`, `cn`, `cn_d` | `nmr.rates.Rate*` (`cn` reuses `RateCH`) | all (`.mf` expressions) | generic as-is (TBC Phase 3) |
| `max` | builtin | all | — |
| `pop_2st` | `models.constraints.pop_2st` | 2st_eyring, 2st_hd, 2st_monomer_{dimer,trimer,tetramer} | — |
| `pop_3st` | `models.constraints.pop_3st` | 3st_eyring*, 3st_binding_partner_2st, 3st_double_binding, 3st_monomer_dimer_{trimer,tetramer} | — |
| `pop_4st` | `models.constraints.pop_4st` | 4st_eyring | — |
| `eyring_rate` | `_eyring.calculate_rate_component` | 2st/3st/3st_linear/3st_fork/4st _eyring | — |
| `kij_2st_eyring`, `pop_2st_eyring` | `settings_2st_eyring` | 2st_eyring | — |
| `kij_3st_eyring`, `kij_3st_eyring_fork`, `pop_3st_eyring` | `settings_3st_eyring` | 3st_eyring, 3st_eyring_linear, 3st_eyring_fork | — |
| `kij_4st_eyring`, `pop_4st_eyring` | `settings_4st_eyring` | 4st_eyring | — |
| `calc_conc`, `populations`, `rates` | `settings_2st_binding` | 2st_binding | — |
| `calc_conc`, `populations`, `rates` | `settings_3st_binding_2st_partner` | 3st_binding_partner_2st | — |
| `calc_conc`, `populations`, `rates` | `settings_3st_double_binding` | 3st_double_binding | — |
| `calc_conc`, `populations`, `rates` | `settings_4st_binding_2st_partner` | 4st_binding_partner_2st | — |
| `concentrations`, `populations`, `rates` | `settings_4st_binding_3_bound_states` | 4st_binding_3_bound_states | — |
| `equilibrium`, `populations`, `conformational_rates`, `binding_rates`, `intrinsic_values`, `kon_values` | `settings_3st_binding_cs` / `settings_3st_binding_if` | 3st_binding_cs, 3st_binding_if | — |
| `concentrations`, `populations`, `rates` | `settings_2st_monomer_{dimer,trimer,tetramer}` | those three models | — |
| `concentrations` / `concetrations` (sic, upstream spelling), `populations`, `rates` | `settings_3st_monomer_dimer_{tetramer,trimer}` | those two models | — |
| `population_complement`, `pair_rates` | `settings_nst` | 3st…6st, `_linear`, `_fork`, `3st_triangle` | — |

Models with no user functions: `1st`, `2st`, `4st_hd` (only `max` and the rate
functions). Registered models (34): `1st 2st 2st_binding 2st_eyring 2st_hd
2st_monomer_dimer 2st_monomer_tetramer 2st_monomer_trimer 3st 3st_binding_cs
3st_binding_if 3st_binding_partner_2st 3st_double_binding 3st_eyring
3st_eyring_fork 3st_eyring_linear 3st_fork 3st_linear
3st_monomer_dimer_tetramer 3st_monomer_dimer_trimer 3st_triangle 4st
4st_binding_3_bound_states 4st_binding_partner_2st 4st_eyring 4st_fork 4st_hd
4st_linear 5st 5st_fork 5st_linear 6st 6st_fork 6st_linear`, each with the
`.mf`, `.rs`, `.tc` extensions where `ModelSpec` allows them.

### 5.5 Residual construction (Phase 4 target)

Native fitting does **not** call `Data.scale`. `evaluation/native.py:280-317`
(`_weighted_reduction`, `_normalization_factor`) computes
`Σ (c/e)(x/e) / Σ (c/e)²` as a left-to-right binary64 loop **without** the
`+ eps` that `Data.scale` (`containers/data.py:266-291`) adds to the error, and
returns `None` (an invalid trial) on non-finite values. Residuals are
`(scale·calc − exp)/err` on retained (masked) points
(`evaluation/native.py:903-915`). `compile_residuals` will mirror the
**native** contract; `Data.scale` is only used for publication.

## 6. Architecture — refinements to the brief

### 6.1 Backend object
`src/chemex/backend/{__init__,base,numpy_backend}.py` (NumPy-only imports) and
`src/chemex/backend/jax_backend.py` (imported lazily; enables
`jax_enable_x64`). Protocol as in the brief: `name`, `xp`,
`propagators(liouv, delays, *, dephasing)`, `matrix_power`, `stack`,
`detect`, `evaluate_scientific_function`. Backends are stateless singletons
that pickle and deep-copy **by name** (`__reduce__` → `get_backend(name)`),
because spectrometers are deep-copied (`finalize_native_construction`,
`new_native_workspace`, MC/BS profiles) and pickled to worker processes.

### 6.2 Where the backend lives and how JAX evaluation is isolated
The backend is an attribute of `ISLiouvillianEngine` (default NumPy);
`Spectrometer.backend`/`Spectrometer.xp` delegate to it. JAX evaluation never
switches a shared spectrometer: `chemex.jax` deep-copies the profile's
spectrometer **inside each trace** and sets the JAX backend on the copy. The
copy (and the pulse caches holding tracers) dies with the trace, so nothing
traced outlives the call, and two threads can trace the same profile
concurrently. No module-global switch and no context variable are needed; a
`contextvars` scope can be added later if a use case appears. The native
fingerprint/descriptor (`Spectrometer._build_native_kernel_descriptor`) is not
touched, so evaluator identities are unchanged.

### 6.3 Prefer namespace-neutral operators
Where an operator form is bit-identical on NumPy, use it instead of threading
`xp` through: `reduce(operator.matmul, …)` instead of `reduce(np.matmul, …)`,
`a.mean(axis=0)` instead of `np.mean(a, axis=0)`, `np.array(index_list)`
for fancy indexing. This keeps the experiment diffs to a handful of lines.

### 6.4 Phase-1 scope
The seven "no experiment change" examples still end in `np.array([...])` and
(five of them) `numpy.linalg.matrix_power`; the prototype only handled these
by patching `np`. Without patching they fail under `jit`. Phase 1 therefore
also ports those six modules (`cest_13c`, `cpmg_15n_ip`, `cpmg_chd2_1h_ap`,
`dcest_15n`, `relaxation_hznz`, `relaxation_nz`), two lines each.

### 6.5 Gradient tests before Phase 3
Phase 2 gradient tests differentiate with respect to the *local* spectrometer
parameter values (`profile.name_map` targets), perturbed directly in the
NumPy reference. This tests the NMR/experiment layer independently of the
constraint program; Phase 3 adds end-to-end gradients through the program.

### 6.6 Oligomerization root solve without new dependencies
Proposal: a hand-written bracketed Newton/bisection (`lax.while_loop`) with an
implicit-function-theorem `jax.custom_jvp`, instead of adding `optimistix`.
(Adding a dependency needs approval.)

### 6.7 Fork files
`AGENTS.md` already exists upstream and is the canonical agent guide; the
fork appends a clearly marked "ChemEx-JAX fork" section instead of replacing
it, and `CLAUDE.md` points to `AGENTS.md`.

## 6.8 Phase 1 — backend abstraction in the NMR engine (done)

What changed:

| File | Change |
| --- | --- |
| `src/chemex/backend/{__init__,base,numpy_backend}.py` (new) | `Backend` protocol, `get_backend`, NumPy singleton delegating verbatim (`calculate_propagators`, `numpy.linalg.matrix_power`, `np.array`, `detect_signal`); pickles/deep-copies by name; lazy `chemex.nmr` imports keep the package a leaf |
| `src/chemex/backend/jax_backend.py` (new) | JAX kernels: float64 guard, corrected `expm` (below), Daleckii-Krein `custom_jvp` dephased propagator (ported from the prototype), backend `detect` returning a 0-d array |
| `nmr/_engine/engine.py` | `backend`/`xp` attributes (default NumPy; setter bumps the free-evolution generation so pulse caches rebuild); `detect` passes the backend |
| `nmr/_engine/readout.py` | `detect(..., backend=NUMPY_BACKEND)` delegates to `backend.detect` |
| `nmr/_engine/magnetization.py` | `m += x` → `m = m + x` (same ufunc and operand order) |
| `nmr/_pulses/kernel.py` | propagators and phase stacking via `engine.backend`; `reduce(operator.matmul, …)` |
| `nmr/_pulses/library.py` | `p90_i[[3,0,1,2]]` → module-level index arrays |
| `nmr/spectrometer.py` | `backend`, `xp`, `with_backend()` (private deep copy) |
| six catalog modules | `return np.array(...)` → `spectrometer.backend.stack(...)`, `matrix_power` → `spectrometer.backend.matrix_power` |

**JAX `expm` bug (found in Phase 1).** `jax.scipy.linalg.expm` (0.11.2,
`jax/_src/scipy/linalg.py::_calc_P_Q`) uses
`n_squarings = floor(log2(|A|_1 / theta_13))`; Higham (2005) — and JAX's own
docstring — require `ceil`. The scaled norm can then lie in
`[theta_13, 2 theta_13)`, outside Padé-13's accuracy bound. For the D-CEST
free-precession delay of DCEST_15N_HD_EXCH residue 159N (6x6,
`|A|_1 = 19.4`) JAX returns `exp(A)` with 2.1e-9 relative error against a
40-digit mpmath reference (SciPy: 1.4e-14); the 120-fold DANTE matrix power
turns that into a 2.2e-9 profile error, failing the 1e-9 parity criterion.
The prototype's larger forward errors (1e-12 – 1e-9) have the same cause.
Fix: `jax_backend.expm` scales with `ceil` itself, calls JAX's Padé step with
`max_squarings=0` on the scaled matrix, squares with a `lax.scan`/`lax.cond`
loop (≤ 20 squarings; JAX defaults to 16), and returns NaN beyond the squaring budget
instead of a silently wrong matrix. Tested against SciPy for norms 0 – 1e3
(`tests/backend/test_jax_expm.py`). Worth reporting upstream to JAX.

Forward parity under `jax.jit` (3 profiles spread over each example; local
spectrometer values traced; tolerance 1e-9):

| Example | Model | Profiles | Max rel. error (3 profiles) |
| --- | --- | --- | --- |
| CEST_13C | 2st | 16 | 1.4e-13 |
| CPMG_15N_IP | 2st | 108 | 4.5e-14 |
| CPMG_CHD2_1H_AP | 2st | 32 | 9.3e-14 |
| DCEST_15N | 2st | 108 | 3.2e-14 |
| DCEST_15N_HD_EXCH | 2st_hd | 462 | 2.9e-14 |
| RELAXATION_HZNZ | 2st | 5 | 1.4e-15 |
| RELAXATION_NZ | 2st | 5 | 1.2e-16 |

Also tested: JAX (`jit`, `jacfwd`, `vmap`) then NumPy in one process leaves
the NumPy profile bit-identical and the shared spectrometer untouched;
float32 Liouvillians are rejected; `import chemex`/CLI/plugin registration
never import JAX; backends survive deepcopy/pickle.

Tooling note: until the `jax` extra exists (Phase 5), `ty check` reports the
three unresolved `jax` imports in `jax_backend.py`; everything else is clean.
`prototypes/` is excluded locally via `.git/info/exclude` (so `ruff check .`
and `git status` ignore it) and from pytest via the root `conftest.py`.

## 6.9 Phase 2 — experiment catalog (done)

What changed (NumPy outputs byte-identical: golden compare, 38 runs / 500
files, 0 differences):

| Area | Change |
| --- | --- |
| all catalog modules (34 + 3 WIP) | `return np.array([...])` → `spectrometer.backend.stack(...)`; `numpy.linalg.matrix_power` → `spectrometer.backend.matrix_power`; `reduce(np.matmul, …)` → `reduce(operator.matmul, …)`; `np.mean(x, axis=0)` → `x.mean(axis=0)`; `np.stack` → backend stack |
| list indexing | `p[[i, j]]` → `p[np.array([i, j])]` (all sites in §5.2, incl. the 2-D `p9024090_nh_*[[0, 1], [0, 1]]`) |
| `shift_*` | `_find_nearest(..., xp)` (default NumPy, unchanged) |
| `wip/relaxation_15n_r1rho_eig` | `spectrometer.xp.exp` |
| `nmr/_engine/analysis.py` | `xp.linalg.eigvals`; traceable masked max for R1rho (NaN instead of raising when no real eigenvalue; NumPy path unchanged) |
| `nmr/_engine/effective_field.py` | `xp` argument: traced `arctan2`, functional `.at[].set` tilt (NumPy path unchanged) |
| `nmr/_engine/views.py` | `np.squeeze(x)` → `x.squeeze()` |
| `backend/jax_backend.py` | squaring budget raised 16 → 20 (|A|_1 ≤ 5.6e6; largest example needs 1.3e4, see below) |

Golden coverage fix: `examples/Experiments/DCEST_15N_3States` has no
`run.sh`, only four `run_*.sh` variants (3st_fork / 3st_linear, with and
without DRD-CEST). The golden tool now runs every `run*.sh` of a directory
without `run.sh`; the four new manifest entries (48 files) were generated on a
`d0ba6e34` worktree and are deterministic there (two runs, 0 differences).

**Synthetic configurations** (`tests/backend/synthetic/`, generated by
`_generate.py`) cover the 8 registered types without a shipped example:
`coscest_13c`, `dcest_13c`, `noesyfpgpph19`, `shift_13c_sq`, `shift_15n_sq`,
`cest_15n_test`, `wip.relaxation_15n_r1rho`, `wip.relaxation_15n_r1rho_eig`.
Settings follow `website/docs/experiments/**` (or the settings class when no
page exists); spin systems and values come from the closest shipped example.
All 37 registered experiment types are therefore covered (46 cases).

**Propagator norms.** Largest `|L t|_1` of a non-dephased propagator over
all profiles of every example: 1.3e4 (COSCEST_1HN_IP_AP; 12 squarings),
then CEST_1HN_IP_AP 1.1e4, 2stBinding 5.5e3, CEST_13C 5.3e3.

**Gradient finding: the FD reference, not AD, limits the brief's gradient
criterion.** `jacfwd` vs Richardson FD of NumPy (best of steps 1e-3/1e-4/1e-5)
fails 1e-5 for a few parameters with effect 1e-6 – 1e-4 (cross-relaxation,
auxiliary relaxation rates: `etaz_*`, `etaxy_*`, `r1a_is_*`, …). FD of a
float64 function has a noise floor of ~`eval_noise / (effect * step)`; the
NumPy eig/solve path has eval noise up to ~5e-12 (CEST_1HN_IP_AP). A 30–34
digit mpmath re-evaluation of the *unmodified* sequence code
(`tests/backend/_mp_backend.py`) gives exact derivatives:

| Example | Parameter (effect) | NumPy FD vs exact | `jacfwd` vs exact |
| --- | --- | --- | --- |
| CPMG_13CO_AP | etaxy_i_b (1.4e-6) | 1.5e-5 | 1.4e-14 |
| CPMG_13CO_AP | etaz_i_a (2.0e-6) | 1.1e-5 | 1.6e-14 |
| CPMG_CH3_13C_H2C | r1_i_b (2.2e-6) | 3.2e-5 | 1.7e-14 |
| CPMG_CH3_13C_H2C | etaz_i_a (3.2e-6) | 1.3e-5 | 1.7e-14 |
| CPMG_CH3_13C_H2C | r1a_is_b (4.7e-6) | 1.5e-5 | 1.5e-14 |
| CEST_1HN_IP_AP (dephasing) | etaz_i_a (1.1e-4) | 1.0e-4 | 3.6e-12 |
| CEST_1HN_IP_AP (dephasing) | etaz_i_b (2.4e-6) | 3.3e-3 | 4.1e-12 |
| CEST_1HN_IP_AP (dephasing) | r1a_is_b (2.8e-3) | 5.9e-6 | 1.8e-13 |
| CPMG_HN_DQ_ZQ | r2_i_a (1.1e-5) | 2.1e-5 | 5.0e-14 |
| CPMG_HN_DQ_ZQ | r1_s_a (7.6e-6) | 2.3e-5 | 2.7e-14 |
| CPMG_HN_DQ_ZQ | etaxy_s_b (9.4e-6) | 1.6e-5 | 2.2e-15 |
| CEST_15N_CW (8 params, effect 1e-6 – 4e-5) | | 1e-5 – 3e-4 | 1.6e-12 – 4.6e-11 |

The test therefore keeps the brief's criterion and, only where NumPy FD
misses 1e-5 for a parameter with effect ≥ 1e-6, falls back to the exact
reference with the *stricter* tolerance 1e-9 (on the 8 most sensitive points
plus an even spread; the subset evaluation is verified against the full
profile in float64). This needs `mpmath` at test time — **open question**.

**Dephasing `custom_jvp` at degenerate points** (CEST_15N Liouvillian,
on-resonance + B1; JVP along each of 7 parameter directions vs exact mpmath
derivative): kex = 0 ≤ 2e-14; pB = 0 ≤ 2e-12; Δω = 0 ≤ 3e-13; identical
R1/R2 ≤ 1e-13; all at once (exactly repeated eigenvalues) ≤ 2e-15.

**Transformations.** Every case: `jit`; `vmap` over a batch of parameter
vectors (matches per-vector `jit` to 1e-12); `jit(value_and_grad(χ²))`
finite. `vmap` *across residues* needs per-residue spectrometer constants
(e.g. CEST_13C_LABEL_CN J-multiplets differ per carbon), i.e. the
shape-grouped compilation of Phase 4, and is tested there.

**Coverage (all 46 cases pass; one profile for gradients, three for forward).**
"FD" is the worst NumPy-Richardson error over strong parameters that pass the
brief's FD criterion; "exact (n)" is the worst error against the 30-digit
reference for the n strong parameters whose NumPy FD was noise-limited. Weak
parameters are reported by `tests/backend/report_parity.py`.

| Case | Model | Experiment types | Fwd (3 prof.) | Worst strong grad: FD / exact (n) | jacfwd vs jacrev | Weak params (effect < 1e-6) |
| --- | --- | --- | --- | --- | --- | --- |
| Combinations/2stBinding | 2st_binding | cest_15n, cpmg_15n_ip | 2.0e-13 | 8.7e-07 | 3.3e-16 | 0 |
| Combinations/CPMG_CH3_1H_DQ_TQ | 2st | cpmg_ch3_1h_dq, cpmg_ch3_1h_tq | 7.3e-14 | 7.1e-06 / 1.2e-12 (2 exact) | 1.9e-14 | 2 |
| Combinations/N15_NH_RDC | 2st | cpmg_15n_ip, cpmg_15n_tr | 3.6e-14 | 7.6e-07 | 2.4e-15 | 0 |
| Combinations/Shifts | 2st | cpmg_15n_ip, cpmg_1hn_ap, shift_15n_sqmq | 3.1e-11 | 4.0e-07 | 9.6e-16 | 0 |
| Experiments/CEST_13C | 2st | cest_13c | 1.4e-13 | 7.0e-09 | 3.2e-15 | 0 |
| Experiments/CEST_13C_LABEL_CN | 2st | cest_13c | 1.2e-14 | 1.2e-09 | 3.1e-16 | 0 |
| Experiments/CEST_15N | 2st | cest_15n | 2.1e-14 | 4.3e-09 | 1.7e-16 | 0 |
| Experiments/CEST_15N_CW | 2st.mf | cest_15n_cw | 3.8e-13 | 2.5e-06 / 4.6e-11 (8 exact) | 2.7e-16 | 4 |
| Experiments/CEST_15N_LABEL_CN | 2st | cest_15n | 7.8e-15 | 9.2e-10 | 2.1e-16 | 0 |
| Experiments/CEST_15N_TR | 2st.mf | cest_15n_tr | 1.5e-13 | 6.6e-08 | 1.1e-16 | 0 |
| Experiments/CEST_1HN_AP | 2st | cest_1hn_ap | 6.5e-14 | 1.3e-06 | 2.0e-16 | 0 |
| Experiments/CEST_1HN_IP_AP | 2st.rs | cest_1hn_ip_ap | 1.0e-11 | 5.9e-06 / 1.6e-11 (2 exact) | 2.3e-16 | 0 |
| Experiments/CEST_CH3_1H_IP_AP | 2st.rs | cest_ch3_1h_ip_ap | 7.9e-12 | 6.8e-06 / 2.0e-12 (2 exact) | 2.5e-16 | 0 |
| Experiments/COSCEST_1HN_IP_AP | 2st.rs | coscest_1hn_ip_ap | 8.4e-13 | 2.1e-06 / 2.9e-12 (1 exact) | 7.9e-16 | 0 |
| Experiments/CPMG_13C_IP | 2st | cpmg_13c_ip | 8.7e-14 | 1.0e-06 | 4.9e-15 | 0 |
| Experiments/CPMG_13CO_AP | 2st | cpmg_13co_ap | 2.2e-14 | 8.5e-07 / 1.7e-12 (2 exact) | 3.3e-15 | 3 |
| Experiments/CPMG_15N_IP | 2st | cpmg_15n_ip | 4.5e-14 | 2.4e-07 | 1.9e-15 | 0 |
| Experiments/CPMG_15N_IP_0013 | 2st | cpmg_15n_ip_0013 | 4.8e-14 | 1.4e-07 | 2.4e-15 | 0 |
| Experiments/CPMG_15N_TR | 2st | cpmg_15n_tr | 8.7e-14 | 6.0e-07 | 1.8e-15 | 0 |
| Experiments/CPMG_15N_TR_0013 | 2st | cpmg_15n_tr_0013 | 4.1e-14 | 4.6e-07 | 2.1e-15 | 0 |
| Experiments/CPMG_1HN_AP | 2st | cpmg_1hn_ap | 1.0e-13 | 2.3e-06 / 2.4e-12 (1 exact) | 1.4e-14 | 2 |
| Experiments/CPMG_1HN_AP_0013 | 2st | cpmg_1hn_ap_0013 | 1.8e-14 | 2.7e-07 / 2.2e-13 (1 exact) | 4.5e-15 | 2 |
| Experiments/CPMG_CH3_13C_H2C | 2st | cpmg_ch3_13c_h2c | 4.4e-14 | 3.6e-06 / 3.0e-12 (3 exact) | 4.8e-15 | 2 |
| Experiments/CPMG_CH3_13C_H2C_0013 | 2st | cpmg_ch3_13c_h2c_0013 | 3.0e-14 | 3.0e-06 | 6.1e-15 | 1 |
| Experiments/CPMG_CH3_1H_SQ | 2st | cpmg_ch3_1h_sq | 2.8e-13 | 8.1e-06 / 3.3e-12 (1 exact) | 8.5e-15 | 2 |
| Experiments/CPMG_CH3_1H_TQ | 2st | cpmg_ch3_1h_tq | 1.0e-13 | 3.5e-06 / 4.2e-12 (3 exact) | 2.8e-14 | 2 |
| Experiments/CPMG_CH3_1H_TQ_DIFF | 2st | cpmg_ch3_1h_tq_diff | 2.1e-14 | 4.8e-06 / 2.7e-13 (1 exact) | 3.0e-14 | 1 |
| Experiments/CPMG_CH3_MQ | 2st | cpmg_ch3_mq | 3.8e-14 | 1.2e-09 | 1.0e-17 | 2 |
| Experiments/CPMG_CHD2_1H_AP | 2st | cpmg_chd2_1h_ap | 9.3e-14 | 7.2e-06 / 7.8e-12 (1 exact) | 8.3e-15 | 3 |
| Experiments/CPMG_HN_DQ_ZQ | 2st | cpmg_hn_dq_zq | 1.3e-13 | 7.4e-06 / 1.2e-12 (3 exact) | 2.5e-14 | 10 |
| Experiments/DCEST_15N | 2st | dcest_15n | 3.2e-14 | 4.4e-07 | 7.7e-16 | 0 |
| Experiments/DCEST_15N_3States/run_FIFU | 3st_fork | dcest_15n | 1.5e-13 | 3.6e-06 | 6.5e-16 | 0 |
| Experiments/DCEST_15N_3States/run_FIFU_drd | 3st_fork | dcest_15n | 1.3e-13 | 3.6e-06 | 6.5e-16 | 0 |
| Experiments/DCEST_15N_3States/run_FIU | 3st_linear | dcest_15n | 1.4e-13 | 4.7e-06 | 8.4e-16 | 0 |
| Experiments/DCEST_15N_3States/run_FIU_drd | 3st_linear | dcest_15n | 1.3e-13 | 4.7e-06 | 8.4e-16 | 0 |
| Experiments/DCEST_15N_HD_EXCH | 2st_hd | dcest_15n | 2.9e-14 | 2.7e-07 | 1.1e-15 | 0 |
| Experiments/RELAXATION_HZNZ | 2st | relaxation_hznz | 1.4e-15 | 4.6e-09 | 1.7e-17 | 0 |
| Experiments/RELAXATION_NZ | 2st | relaxation_nz | 1.2e-16 | 5.3e-13 | 3.0e-17 | 0 |
| synthetic/CEST_15N_TEST | 2st | cest_15n_test | 5.8e-14 | 6.9e-07 | 8.7e-16 | 0 |
| synthetic/COSCEST_13C | 2st | coscest_13c | 3.1e-13 | 1.4e-08 | 4.0e-15 | 0 |
| synthetic/DCEST_13C | 2st | dcest_13c | 9.0e-14 | 7.6e-09 | 1.9e-15 | 0 |
| synthetic/NOESYFPGPPH19 | 2st | noesyfpgpph19 | 3.2e-16 | 3.8e-12 | 6.9e-18 | 0 |
| synthetic/RELAXATION_15N_R1RHO | 2st | wip.relaxation_15n_r1rho | 1.9e-14 | 2.0e-08 | 1.1e-15 | 0 |
| synthetic/RELAXATION_15N_R1RHO_EIG | 2st | wip.relaxation_15n_r1rho_eig | 2.2e-15 | 8.7e-09 | 1.8e-16 | 0 |
| synthetic/SHIFT_13C_SQ | 2st | shift_13c_sq | 1.3e-16 | 4.8e-07 | 5.3e-18 | 0 |
| synthetic/SHIFT_15N_SQ | 2st | shift_15n_sq | 1.2e-16 | 6.1e-07 | 1.1e-16 | 0 |

Experiment-type coverage (37 registered types):

| Experiment type | Covered by |
| --- | --- |
| `cest_13c` | Experiments/CEST_13C, Experiments/CEST_13C_LABEL_CN |
| `cest_15n` | Combinations/2stBinding, Experiments/CEST_15N, Experiments/CEST_15N_LABEL_CN |
| `cest_15n_cw` | Experiments/CEST_15N_CW |
| `cest_15n_test` | synthetic/CEST_15N_TEST |
| `cest_15n_tr` | Experiments/CEST_15N_TR |
| `cest_1hn_ap` | Experiments/CEST_1HN_AP |
| `cest_1hn_ip_ap` | Experiments/CEST_1HN_IP_AP |
| `cest_ch3_1h_ip_ap` | Experiments/CEST_CH3_1H_IP_AP |
| `coscest_13c` | synthetic/COSCEST_13C |
| `coscest_1hn_ip_ap` | Experiments/COSCEST_1HN_IP_AP |
| `cpmg_13c_ip` | Experiments/CPMG_13C_IP |
| `cpmg_13co_ap` | Experiments/CPMG_13CO_AP |
| `cpmg_15n_ip` | Combinations/2stBinding, Combinations/N15_NH_RDC, Combinations/Shifts, Experiments/CPMG_15N_IP |
| `cpmg_15n_ip_0013` | Experiments/CPMG_15N_IP_0013 |
| `cpmg_15n_tr` | Combinations/N15_NH_RDC, Experiments/CPMG_15N_TR |
| `cpmg_15n_tr_0013` | Experiments/CPMG_15N_TR_0013 |
| `cpmg_1hn_ap` | Combinations/Shifts, Experiments/CPMG_1HN_AP |
| `cpmg_1hn_ap_0013` | Experiments/CPMG_1HN_AP_0013 |
| `cpmg_ch3_13c_h2c` | Experiments/CPMG_CH3_13C_H2C |
| `cpmg_ch3_13c_h2c_0013` | Experiments/CPMG_CH3_13C_H2C_0013 |
| `cpmg_ch3_1h_dq` | Combinations/CPMG_CH3_1H_DQ_TQ |
| `cpmg_ch3_1h_sq` | Experiments/CPMG_CH3_1H_SQ |
| `cpmg_ch3_1h_tq` | Combinations/CPMG_CH3_1H_DQ_TQ, Experiments/CPMG_CH3_1H_TQ |
| `cpmg_ch3_1h_tq_diff` | Experiments/CPMG_CH3_1H_TQ_DIFF |
| `cpmg_ch3_mq` | Experiments/CPMG_CH3_MQ |
| `cpmg_chd2_1h_ap` | Experiments/CPMG_CHD2_1H_AP |
| `cpmg_hn_dq_zq` | Experiments/CPMG_HN_DQ_ZQ |
| `dcest_13c` | synthetic/DCEST_13C |
| `dcest_15n` | Experiments/DCEST_15N, Experiments/DCEST_15N_3States/run_FIFU, Experiments/DCEST_15N_3States/run_FIFU_drd, Experiments/DCEST_15N_3States/run_FIU, Experiments/DCEST_15N_3States/run_FIU_drd, Experiments/DCEST_15N_HD_EXCH |
| `noesyfpgpph19` | synthetic/NOESYFPGPPH19 |
| `relaxation_hznz` | Experiments/RELAXATION_HZNZ |
| `relaxation_nz` | Experiments/RELAXATION_NZ |
| `shift_13c_sq` | synthetic/SHIFT_13C_SQ |
| `shift_15n_sq` | synthetic/SHIFT_15N_SQ |
| `shift_15n_sqmq` | Combinations/Shifts |
| `wip.relaxation_15n_r1rho` | synthetic/RELAXATION_15N_R1RHO |
| `wip.relaxation_15n_r1rho_eig` | synthetic/RELAXATION_15N_R1RHO_EIG |

**Test runtime.** Compilation dominates: COSCEST_1HN_IP_AP takes ~110 s per
`jit` compile (Python loops over offsets × shaped-pulse slices are unrolled).
The full backend suite took 60 min on 10 workers before the exact reference
was restricted to a row subset (CEST_15N_CW gradients: 55 min → 6 min); the
46 gradient tests now take 19.5 min on 10 workers.

**Intermittent upstream failure (not reproduced).** One full-suite run
(`-n 10`) failed `tests/test_native_resampling.py::test_serial_and_reordered_multi_worker_execution_use_fresh_single_owners`
(traceback not captured). It passed in isolation (3/3), in a second full run,
and in 60 stress runs at 10-way concurrency (30 on `d0ba6e34`, 30 on this
branch). The test synchronises 4 replicate threads with 5 s wall-clock waits;
the resampling/threading code is untouched by the port.

## 7. Decisions (maintainer, 2026-10-02)

1. Golden comparisons exclude every `*identity` value (see §2).
2. `compile_residuals` uses the **native** fit scaling
   (`evaluation/native.py::_normalization_factor`, no `+eps`), not
   `Data.scale`.
3. Phase 1 also ports the six modules behind the seven "no-change" examples
   (`cest_13c`, `cpmg_15n_ip`, `cpmg_chd2_1h_ap`, `dcest_15n`,
   `relaxation_hznz`, `relaxation_nz`).
4. The oligomerization root solve is hand-written (bracketed Newton +
   implicit-function `custom_jvp`); no `optimistix` dependency.
5. `prototypes/` is not committed (kept locally as a reference/oracle).

## 8. Open questions (Phase 0, resolved above unless noted)


1. **Phase-1 scope (§6.4).** OK to port the six modules behind the seven
   "no-change" examples in Phase 1 (`np.array` → `backend.stack`,
   `matrix_power` → `backend.matrix_power`), so Phase-1 parity runs under
   real `jit` rather than eager JAX?
2. **Residual target (§5.5).** The brief names `Data.scale`; native fitting
   uses `evaluation/native.py::_normalization_factor` (no `+eps`). Proposal:
   `compile_residuals` matches the native evaluator (what TRF actually
   minimises) and the tests also report the difference to `Data.scale`.
3. **Oligomerization solver (§6.6).** Hand-written bracketed Newton +
   implicit `custom_jvp` (no new dependency) instead of `optimistix`?
4. **Prototype gradient MISMATCHes (§4).** Four examples exceed the
   prototype's 1e-5 gradient threshold (1.9e-5 – 5.6e-5). Forward parity is
   fine and the values move between runs, which points to FD noise in weak
   parameters; Phase 2 will diagnose per parameter with the brief's
   effect-weighted criterion and report, not loosen anything.
5. **Combinations examples.** The prototype only covered
   `examples/Experiments/*`. Phase 2 forward parity will also cover
   `examples/Combinations/*` (2stBinding needs Phase-3 twins, so its JAX
   parity lands in Phase 3; Shifts uses `shift_*` eigenvalue experiments).
6. **GPU.** An NVIDIA GPU is present on the dev machine but only CPU jaxlib is
   installed. GPU checks of the `expm` path (and the documented CPU-only
   `eig` path) can be run in Phase 5 if a CUDA jaxlib is wanted.
