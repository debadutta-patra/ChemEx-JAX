# ChemEx: NMR Chemical Exchange Analysis Tool

> [!IMPORTANT]
> **This is ChemEx-JAX, a modified fork of
> [ChemEx](https://github.com/gbouvignies/ChemEx) by Guillaume Bouvignies.**
> It adds an optional [JAX](https://docs.jax.dev) backend that runs ChemEx's
> experiments, kinetic models and parameter constraints as differentiable,
> compilable functions, for exact Jacobians, Fisher information, experiment
> design, gradient-based sampling and fast batch simulation. Everything else
> is upstream ChemEx: `chemex fit`, the TOML formats, models and optimizers
> are unchanged, and the default NumPy results are byte-identical to upstream.
> See [Optional JAX backend](#optional-jax-backend-chemex-jax-fork) and the
> user guide page `website/docs/user_guide/jax_backend.md`. Please report
> issues with the JAX backend to this fork, not to upstream ChemEx.

[![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64.svg?logo=ruff)](https://docs.astral.sh/ruff/)

## Table of Contents

- [ChemEx: NMR Chemical Exchange Analysis Tool](#chemex-nmr-chemical-exchange-analysis-tool)
    - [Table of Contents](#table-of-contents)
    - [About ChemEx](#about-chemex)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
        - [Recommended installation](#recommended-installation)
        - [Try ChemEx without installing it](#try-chemex-without-installing-it)
        - [Updating ChemEx](#updating-chemex)
        - [Reproducible, version-pinned installation](#reproducible-version-pinned-installation)
        - [Alternative: pip](#alternative-pip)
        - [Conda packages](#conda-packages)
        - [Optional JAX backend (ChemEx-JAX fork)](#optional-jax-backend-chemex-jax-fork)
    - [Contributing](#contributing)
    - [Support and Documentation](#support-and-documentation)
    - [License](#license)

<!-- -   [Citing ChemEx](#citing-chemex) -->

## About ChemEx

ChemEx is an advanced, open-source software specifically designed for analyzing NMR experimental data to characterize chemical exchange processes. Ideal for researchers and scientists in the field of biochemistry and molecular biology, ChemEx aids in the analysis of NMR experiments like Carr-Purcell-Meiboom-Gill (CPMG) relaxation dispersion and Chemical Exchange Saturation Transfer (CEST).

## Prerequisites

ChemEx requires **Python 3.13 or later**. When a requested ChemEx release needs
a newer Python, uv can use a compatible installed interpreter or download a
managed one when automatic Python downloads are enabled (the default).

## Installation

[PyPI](https://pypi.org/project/chemex/) is the authoritative Python package
distribution for ChemEx.

### Recommended installation

[uv](https://docs.astral.sh/uv/) installs ChemEx as an application in an
isolated environment. On macOS, install uv with Homebrew:

```shell
brew install uv
```

On Linux or Windows, follow Astral's current
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Then install ChemEx from PyPI and verify that it starts:

```shell
uv tool install chemex@latest
chemex --version
```

Requesting `@latest` prevents uv from selecting an older ChemEx release to match
an outdated Python interpreter.

### Try ChemEx without installing it

```shell
uvx chemex@latest --help
```

### Updating ChemEx

Update an unpinned tool installation with:

```shell
uv tool upgrade chemex
```

### Reproducible, version-pinned installation

To install a specific release:

```shell
uv tool install "chemex==2026.09.2"
```

### Alternative: pip

If you need conventional Python tooling, use Python 3.13 or later to install
ChemEx from PyPI inside a virtual environment:

```shell
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install chemex
```

### Conda packages

The historical conda-forge package is no longer maintained by the ChemEx
project and may be outdated. Install the current PyPI release with uv or pip.

### Optional JAX backend (ChemEx-JAX fork)

The JAX backend is an optional extra of this fork (it is not in the `chemex`
package on PyPI). Install the fork with the extra from its Git repository:

```shell
python -m pip install "chemex[jax] @ git+https://github.com/debadutta-patra/ChemEx-JAX@jax-backend"
# or, in a development checkout:
uv sync --extra jax
```

ChemEx itself (`import chemex`, the `chemex` command) works without JAX. To
check the backend on your own data, run

```shell
chemex compare-backends -e Experiments/*.toml -p Parameters/parameters.toml
```

and see the user guide page "Using the JAX backend" for the Python API
(`chemex.jax`), worked examples (`examples/jax/`), and its costs (compile
time, memory, float64 only).

## Contributing

We encourage contributions from the community. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to make ChemEx better. For any issues or suggestions, please open an issue or a discussion on our [GitHub repository](https://github.com/gbouvignies/ChemEx).

## Support and Documentation

For additional support, tutorials, and detailed documentation, visit the [ChemEx Documentation](https://gbouvignies.github.io/ChemEx/).

## License

ChemEx is licensed under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See the [LICENSE](LICENSE.md) file for more details.

ChemEx-JAX is a modified version of ChemEx and is distributed under the same
licence, GPL-3.0-or-later. Upstream copyright notices are retained. Files
changed in the fork carry a "Modified in the ChemEx-JAX fork" comment, and
files added by the fork are marked "ChemEx-JAX fork addition". The fork's
changes are described in `JAX_PORT_NOTES.md` and `CHANGELOG.md`. JAX and
jaxlib are optional dependencies under the Apache License 2.0, which is
compatible with GPL-3.0.

<!-- ## Citing ChemEx

If you use ChemEx in your research, please cite it as follows: [Citation details](#). -->

---

Developed with ❤️ by the [ChemEx Contributors](https://github.com/gbouvignies/ChemEx/graphs/contributors)
