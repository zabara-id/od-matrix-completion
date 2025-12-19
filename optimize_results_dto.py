from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MirrorDescentResult:
    """
    Контейнер с результатами и траекториями зеркального спуска.
    """

    mode: str
    D_final: np.ndarray
    flow_final: np.ndarray
    times_final: np.ndarray
    objective_history: List[float]
    rel_l1_history: List[float]
    kl_history: List[float]
    gradient_sources: List[str]
    switch_iter: int | None = None
