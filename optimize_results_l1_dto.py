from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MirrorDescentL1Result:
    """
    Контейнер с результатами и траекториями зеркального спуска с L1-регуляризацией к D_true.
    """

    mode: str
    D_final: np.ndarray
    flow_final: np.ndarray
    times_final: np.ndarray

    objective_history: List[float]
    rel_l1_ref_history: List[float]
    l1_to_true_history: List[float]
    rel_l1_true_history: List[float]
    gradient_sources: List[str]

    data_history: List[float] = field(default_factory=list)
    data_full_history: List[float] = field(default_factory=list)
    step_history: List[float] = field(default_factory=list)
    ls_trials_history: List[int] = field(default_factory=list)
    accepted_history: List[bool] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    switch_iter: int | None = None

