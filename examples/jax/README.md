# JAX examples (ChemEx-JAX fork)

These examples need the optional JAX backend (`pip install 'chemex[jax]'`,
or `uv run --extra jax` from a development checkout).

## `fisher_cpmg_15n_ip.py`: Fisher information with exact Jacobians

Builds `examples/Experiments/CPMG_15N_IP` with ChemEx's own setup code,
restricted (like the example's method STEP1) to residues 15, 31, 33, 34 and
37. It then compiles ChemEx's native weighted residuals with
`chemex.jax.compile_residuals`, and computes the Fisher information
`F = JᵀWJ`, its eigenvalue spectrum, its condition number and the linearised
standard errors `sqrt(diag(F⁻¹))`.

```sh
# at the initial parameter values
uv run --extra jax python examples/jax/fisher_cpmg_15n_ip.py

# at ChemEx's fitted values (run the example's run.sh first)
uv run --extra jax python examples/jax/fisher_cpmg_15n_ip.py \
    --parameters examples/Experiments/CPMG_15N_IP/Parameters/parameters.toml \
                 examples/Experiments/CPMG_15N_IP/Output/STEP1/Parameters/fitted.toml
```

Evaluated at ChemEx's STEP1 fitted values, the χ² (434.56) and every
standard error agree with the values ChemEx itself reports for that fit,
for example KEX_AB ±6.234 and PB ±8.024e-4. So the exact-Jacobian Fisher
information reproduces ChemEx's native covariance.

Memory: forward-mode Jacobians carry one tangent per free parameter through
every propagator. Take Jacobians with respect to fit-step-sized parameter
sets (tens of parameters), not every independent parameter of a large
example.

## `design_cpmg_15n_ip.py`: optimal design of a CPMG experiment

Local optimal design for one residue of CPMG_15N_IP (500 MHz). It computes
ChemEx's profile at every candidate ncyc (on a copy of the example's
profile; no data needed) and takes one exact Jacobian, augmented with the
fitted intensity scale. From that it predicts the standard errors of KEX_AB
and PB for any choice of points and repeats, and selects 26 points greedily.
It then checks the result against draws from the prior (one `vmap`) and
compares CPMG periods (one compile each).

```sh
uv run --extra jax python examples/jax/design_cpmg_15n_ip.py \
    [--residue 15N] [--points 26] [--noise 0.008] [--nu-max 1000]
```

With the example's own parameters and 0.8% noise, the greedy optimum predicts
SE(KEX) 15.5% and SE(PB) 4.9%, against 23.8% and 9.6% for the example's ncyc
list. These are predictions under the assumed model, parameters and noise;
see the user guide ("Designing experiments") for the caveats.

## `next_experiment.py`: which B0 or B1 to record next

Sequential design: `F_next = F_existing + F_candidate`. The existing data's
Fisher information comes from ChemEx's native weighted residuals at the
current estimates. Each candidate (a copy of the experiment TOML with B0 or B1
changed) is built jointly with the existing data and filled with ChemEx's
noiseless prediction at the existing relative noise. The output is a table of
predicted SE(KEX_AB) and SE(PB) per candidate.

```sh
uv run --extra jax python examples/jax/next_experiment.py cpmg-b0 [--values 600 800 1200]
uv run --extra jax python examples/jax/next_experiment.py cest-b1 [--values 5 10 20 40]
```

Pass `--parameters` with your fitted values, and `--snr-exponent` if you
expect sensitivity to change with B0. The user guide has the results for the
shipped examples and the caveats.
