from dataclasses import dataclass, field
from typing import List, Optional

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
    data_history: List[float] = field(default_factory=list)
    data_full_history: List[float] = field(default_factory=list)
    step_history: List[float] = field(default_factory=list)
    ls_trials_history: List[int] = field(default_factory=list)
    accepted_history: List[bool] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    rel_l1_target_history: Optional[List[float]] = None
    switch_iter: int | None = None
