# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Backend selection works without JAX and never mutates shared spectrometers."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys

import numpy as np
import pytest

from chemex.backend import NUMPY_BACKEND, get_backend
from tests.backend._examples import build_example
from tests.backend.conftest import HAS_JAX


def test_import_chemex_does_not_import_jax() -> None:
    code = (
        "import sys, chemex, chemex.cli, chemex.backend, chemex.nmr; "
        "from chemex.runtime import ensure_plugins_registered; "
        "ensure_plugins_registered(); "
        "assert 'jax' not in sys.modules, 'jax imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_numpy_backend_is_default_singleton() -> None:
    assert get_backend() is NUMPY_BACKEND
    assert get_backend("numpy") is NUMPY_BACKEND
    assert copy.deepcopy(NUMPY_BACKEND) is NUMPY_BACKEND
    assert pickle.loads(pickle.dumps(NUMPY_BACKEND)) is NUMPY_BACKEND  # noqa: S301


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("torch")


@pytest.mark.skipif(HAS_JAX, reason="checks the message shown without JAX")
def test_missing_jax_has_install_hint() -> None:
    with pytest.raises(ImportError, match=r"chemex\[jax\]"):
        get_backend("jax")


def test_spectrometers_default_to_numpy_and_survive_copy_and_pickle() -> None:
    profile = build_example("CPMG_15N_IP").profiles[0]
    spectrometer = profile.spectrometer
    assert spectrometer.backend is NUMPY_BACKEND
    assert spectrometer.xp is np
    assert copy.deepcopy(spectrometer).backend is NUMPY_BACKEND
    restored = pickle.loads(pickle.dumps(spectrometer))  # noqa: S301
    assert restored.backend is NUMPY_BACKEND


def test_with_backend_returns_private_copy() -> None:
    profile = build_example("CPMG_15N_IP").profiles[0]
    original = profile.spectrometer
    other = original.with_backend(NUMPY_BACKEND)
    assert other is not original
    assert other._engine is not original._engine
    assert original.backend is NUMPY_BACKEND
