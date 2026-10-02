# Modified in the ChemEx-JAX fork (GPL-3.0-or-later): detection is delegated
# to the engine's array backend.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemex.backend import NUMPY_BACKEND, Backend
from chemex.nmr._engine.detection import build_detection_vector
from chemex.typing import Array


class LiouvillianReadout:
    """Detection expression state and scalar readout for a Liouvillian."""

    def __init__(self, vectors: Mapping[str, Array]) -> None:
        self._vectors = dict(vectors)
        self._detection = ""
        self._detect_vector: Array = np.array([])

    @property
    def detection(self) -> str:
        return self._detection

    @detection.setter
    def detection(self, value: str) -> None:
        detect_vector = build_detection_vector(value, self._vectors).transpose()
        self._detection = value
        self._detect_vector = detect_vector

    def detect(
        self,
        magnetization: Array,
        weights: Array,
        backend: Backend = NUMPY_BACKEND,
    ) -> Any:
        return backend.detect(self._detect_vector, magnetization, weights)
