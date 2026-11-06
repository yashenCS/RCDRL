import math
import random
from typing import Sequence
import numpy as np
from algorithms.rcdrl.core.utils import filtered_feasible_actions


class SafeFirstPolicy:
    def __init__(self, delta: float, epsilon: float, fallback_top_ratio: float = 0.2):
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.fallback_top_ratio = float(fallback_top_ratio)
        if not 0.0 < self.fallback_top_ratio <= 1.0:
            raise ValueError("fallback_top_ratio must lie in (0, 1].")

    def select_greedy_action(
        self,
        subdeadline_risk_all: Sequence[float],
        q_vm_all: Sequence[float],
        remove_vm=None,
    ) -> int:
        subdeadline_risk_all = np.asarray(subdeadline_risk_all, dtype=np.float32)
        q_vm_all = np.asarray(q_vm_all, dtype=np.float32)
        feasible = filtered_feasible_actions(len(subdeadline_risk_all), remove_vm)
        if not feasible:
            raise ValueError(
                "No feasible action is available for safe-first policy selection."
            )
        safe_actions = [a for a in feasible if subdeadline_risk_all[a] <= self.delta]
        if safe_actions:
            return int(
                min(
                    safe_actions,
                    key=lambda a: (
                        float(q_vm_all[a]),
                        float(subdeadline_risk_all[a]),
                        a,
                    ),
                )
            )
        sorted_by_risk = sorted(
            feasible,
            key=lambda a: (float(subdeadline_risk_all[a]), float(q_vm_all[a]), a),
        )
        relaxed_count = max(1, math.ceil(self.fallback_top_ratio * len(sorted_by_risk)))
        relaxed_safe_actions = sorted_by_risk[:relaxed_count]
        return int(
            min(
                relaxed_safe_actions,
                key=lambda a: (float(q_vm_all[a]), float(subdeadline_risk_all[a]), a),
            )
        )

    def select_behavior_action(
        self,
        subdeadline_risk_all: Sequence[float],
        q_vm_all: Sequence[float],
        remove_vm=None,
        explore: bool = True,
    ) -> int:
        greedy_action = self.select_greedy_action(
            subdeadline_risk_all, q_vm_all, remove_vm=remove_vm
        )
        if not explore or self.epsilon <= 0.0:
            return greedy_action
        feasible = filtered_feasible_actions(len(subdeadline_risk_all), remove_vm)
        if random.random() < self.epsilon:
            alternatives = [a for a in feasible if a != greedy_action]
            if alternatives:
                return int(random.choice(alternatives))
        return greedy_action
