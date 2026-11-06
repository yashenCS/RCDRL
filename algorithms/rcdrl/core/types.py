from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np


@dataclass
class RCDRLState:
    ob: np.ndarray
    dag: Any
    node_id: int
    vm_features: np.ndarray
    gace_dag_node_features: Optional[np.ndarray] = None
    active_wf_features: Optional[np.ndarray] = None
    candidate_action_features: Optional[np.ndarray] = None
    global_features: Optional[np.ndarray] = None
    remove_vm: Optional[int] = None


@dataclass
class Transition:
    state: RCDRLState
    action: int
    reward: float
    step_vm_cost: float
    next_state: Optional[RCDRLState]
    done: bool
    y_deadline_all: List[float]
    y_deadline_feasible_actions: List[int]


@dataclass
class Episode:
    transitions: List[Transition]
    total_reward: float
    total_vm_cost: float
    total_deadline_penalty: float
    miss_deadline_num: int
    deadline_violated: int
    vm_exec_hour: float
    vm_tot_hour: float
    gamma_value: Any
    wf_count: int
    deadline_met_wf_count: int
    deadline_missed_wf_count: int
    norm_lateness_sum_all_wf: float
    norm_lateness_sum_missed_wf: float
    soft_clip_norm_lateness_sum_all_wf: float
    soft_clip_norm_lateness_sum_missed_wf: float


@dataclass
class EvalStats:
    avg_return: float
    avg_vm_exec_hour: float
    avg_vm_tot_hour: float
    avg_vm_cost: float
    avg_deadline_penalty: float
    avg_miss_deadline_num: float
    gamma_per_instance: List[Any]
    wf_count: int
    deadline_met_wf_count: int
    deadline_missed_wf_count: int
    deadline_met_wf_rate: float
    deadline_missed_wf_rate: float
    norm_lateness_mean_all_wf: float
    norm_lateness_mean_missed_wf: float
    soft_clip_norm_lateness_mean_all_wf: float
    soft_clip_norm_lateness_mean_missed_wf: float
    vm_cost_total: float
    vm_cost_mean_per_wf: float
