import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def to_serializable(value: Any):

    if isinstance(value, np.ndarray):
        return [to_serializable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    return value


def filtered_feasible_actions(action_num: int, remove_vm: Optional[int]) -> List[int]:

    feasible = list(range(int(action_num)))
    if remove_vm is not None and int(remove_vm) in feasible:
        feasible.remove(int(remove_vm))
    if not feasible:
        feasible = list(range(int(action_num)))
    return feasible


def _require_instance_axis(name: str, data: np.ndarray) -> Tuple[int, int]:

    if data.ndim < 2:
        raise ValueError(f"{name} must have at least two axes [group, instance, ...], but got {data.shape}.")
    return int(data.shape[0]), int(data.shape[1])


def resolve_env_episode_indices(env, config: dict, mode: str, outer_iter: int, ep_num: int) -> Tuple[int, int]:

    if mode not in {"train", "test"}:
        raise ValueError(f"Unsupported episode mode: {mode!r}")

    if mode == "train":
        data = np.asarray(env.train_Set_setting.trainMatrix)
        num_groups, num_instances = _require_instance_axis("trainMatrix", data)
        episodes_per_iter = int(config.get("evaluation", {}).get("episodes_per_iter", config["env"].get("evalNum", 1)))
        flat_episode_id = outer_iter * episodes_per_iter + ep_num
        group_idx = (flat_episode_id // num_instances) % num_groups
        instance_idx = flat_episode_id % num_instances
        return int(group_idx), int(instance_idx)

    data = np.asarray(env.test_Set_setting.testMatrix)
    _, num_instances = _require_instance_axis("testMatrix", data)
    return 0, int(ep_num % num_instances)


def get_gamma_for_episode(env, config: dict, mode: str, outer_iter: int, ep_num: int):

    env_cfg = config["env"]
    if not env_cfg.get("dynamic_gamma", False):
        return env_cfg["gamma_train"] if mode == "train" else env_cfg["gamma_test"]

    group_idx, instance_idx = resolve_env_episode_indices(env, config, mode, outer_iter, ep_num)

    if mode == "train":
        if env_cfg.get("dynamic_gamma_wf", False):
            return env.train_Set_setting.train_dynamic_gamma_wf[group_idx][instance_idx]
        return env.train_Set_setting.train_dynamic_gamma[group_idx][instance_idx]

    if env_cfg.get("dynamic_gamma_wf", False):
        return env.test_Set_setting.test_dynamic_gamma_wf[group_idx][instance_idx]
    return env.test_Set_setting.test_dynamic_gamma[group_idx][instance_idx]


def aggregate_workflow_lateness_metrics(
    workflow_normalized_lateness: dict,
    workflow_soft_clipped_normalized_lateness: dict,
) -> dict:

    normalized_values = [float(value) for value in workflow_normalized_lateness.values()]
    soft_values = [float(value) for value in workflow_soft_clipped_normalized_lateness.values()]

    if len(normalized_values) != len(soft_values):
        raise RuntimeError(
            "workflow_normalized_lateness and workflow_soft_clipped_normalized_lateness have different sizes: "
            f"{len(normalized_values)} vs {len(soft_values)}."
        )

    wf_count = len(normalized_values)
    deadline_missed_wf_count = sum(1 for value in normalized_values if value > 0.0)
    deadline_met_wf_count = wf_count - deadline_missed_wf_count
    missed_keys = [key for key, value in workflow_normalized_lateness.items() if float(value) > 0.0]

    return {
        "wf_count": int(wf_count),
        "deadline_met_wf_count": int(deadline_met_wf_count),
        "deadline_missed_wf_count": int(deadline_missed_wf_count),
        "norm_lateness_sum_all_wf": float(sum(normalized_values)),
        "norm_lateness_sum_missed_wf": float(sum(float(workflow_normalized_lateness[key]) for key in missed_keys)),
        "soft_clip_norm_lateness_sum_all_wf": float(sum(soft_values)),
        "soft_clip_norm_lateness_sum_missed_wf": float(
            sum(float(workflow_soft_clipped_normalized_lateness[key]) for key in missed_keys)
        ),
    }


@dataclass(frozen=True)
class ActionScore:


    action: int
    vm_type_index: int
    vm_cpu: float
    estimated_start_time: float
    estimated_execution_time: float
    estimated_finish_time: float
    task_sub_deadline: float
    meet_sub_deadline: bool
    incremental_vm_cost: float
    extra_vm_hours: int
    remaining_paid_time_after: float


@dataclass
class EpisodeStats:


    total_reward: float
    total_vm_cost: float
    total_deadline_penalty: float
    miss_deadline_num: int
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
    decision_count: int
    selected_vm_cpu_mean: float
    selected_new_vm_rate: float
    selected_action_meet_subdeadline_rate: float


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
    decision_count: int
    selected_vm_cpu_mean: float
    selected_new_vm_rate: float
    selected_action_meet_subdeadline_rate: float


class FastestNewVMPolicy:


    def __init__(self, deadline_eps: float = 1e-8):
        self.deadline_eps = float(deadline_eps)

    @staticmethod
    def _get_remove_vm(state_dict_list: Optional[dict]) -> Optional[int]:
        if state_dict_list is None:
            return None
        remove_vm = state_dict_list.get("removeVM", None)
        return None if remove_vm is None else int(remove_vm)

    @staticmethod
    def _get_action_num(env, state_dict_list: Optional[dict]) -> int:
        if state_dict_list is not None and "0" in state_dict_list and "state" in state_dict_list["0"]:
            return int(len(state_dict_list["0"]["state"]))
        return int(len(env.vm_queues) + env.dcNum * env.VMtypeNum)

    @staticmethod
    def _task_sub_deadline(env) -> float:
        current_wf = getattr(env, "nextWrf", None)
        current_task = getattr(env, "nextTask", None)
        if current_wf is None or current_task is None:
            raise RuntimeError("Cannot read the current workflow/task from env.nextWrf/env.nextTask.")
        if current_wf not in env.appSubDeadline or current_task not in env.appSubDeadline[current_wf]:
            raise KeyError("The current task sub-deadline has not been initialized in env.appSubDeadline.")
        return float(env.appSubDeadline[current_wf][current_task])

    @staticmethod
    def _fastest_vm_type_index(env) -> int:

        vm_vcpus = list(env.set.dataset.vmVCPU)
        if not vm_vcpus:
            raise ValueError("env.set.dataset.vmVCPU is empty; cannot select the fastest VM type.")
        return int(max(range(len(vm_vcpus)), key=lambda idx: (float(vm_vcpus[idx]), -idx)))

    def _candidate_new_fastest_actions(self, env, state_dict_list: Optional[dict] = None) -> Sequence[int]:

        action_num = self._get_action_num(env, state_dict_list)
        remove_vm = self._get_remove_vm(state_dict_list)
        feasible_actions = set(filtered_feasible_actions(action_num, remove_vm))
        active_vm_num = int(len(env.vm_queues))
        fastest_type_idx = self._fastest_vm_type_index(env)

        actions = []
        for dc_idx in range(int(env.dcNum)):
            action = active_vm_num + dc_idx * int(env.VMtypeNum) + fastest_type_idx
            if action in feasible_actions:
                actions.append(int(action))

        if actions:
            return actions


        fallback = active_vm_num + fastest_type_idx
        if 0 <= fallback < action_num:
            return [int(fallback)]
        raise RuntimeError(
            "No feasible new fastest VM action is available: "
            f"active_vm_num={active_vm_num}, fastest_type_idx={fastest_type_idx}, action_num={action_num}."
        )

    def score_action(self, env, action: int) -> ActionScore:

        action = int(action)
        active_vm_num = int(len(env.vm_queues))
        if action < active_vm_num:
            raise ValueError(f"FASTEST-NEW-VM expects a new-VM action, but got active-VM action {action}.")

        current_wf = getattr(env, "nextWrf", None)
        current_task = getattr(env, "nextTask", None)
        decision_time = getattr(env, "nextTimeStep", None)
        if current_wf is None or current_task is None or decision_time is None:
            raise RuntimeError("Cannot score an action because env.nextWrf, env.nextTask, or env.nextTimeStep is None.")

        diff = action - active_vm_num
        vmtype = int(diff % int(env.VMtypeNum))
        vm_cpu = float(env.set.dataset.vmVCPU[vmtype])
        decision_time = float(decision_time)
        task_process_time = float(current_wf.get_taskProcessTime(current_task))
        exec_time = task_process_time / max(vm_cpu, 1e-12)
        start_time = decision_time
        finish_time = start_time + exec_time
        sub_deadline = self._task_sub_deadline(env)
        pay_interval = float(env.set.VMpayInterval)
        extra_hours = int(math.ceil(max(exec_time, 1e-12) / pay_interval))
        remaining_paid_after = extra_hours * pay_interval - exec_time
        price = float(env.set.dataset.vmPrice[int(vm_cpu)])
        incremental_cost = extra_hours * price

        return ActionScore(
            action=action,
            vm_type_index=vmtype,
            vm_cpu=vm_cpu,
            estimated_start_time=float(start_time),
            estimated_execution_time=float(exec_time),
            estimated_finish_time=float(finish_time),
            task_sub_deadline=float(sub_deadline),
            meet_sub_deadline=bool(finish_time <= sub_deadline + self.deadline_eps),
            incremental_vm_cost=float(incremental_cost),
            extra_vm_hours=int(extra_hours),
            remaining_paid_time_after=float(max(0.0, remaining_paid_after)),
        )

    def score_candidate_actions(self, env, state_dict_list: Optional[dict] = None):

        return [self.score_action(env, action) for action in self._candidate_new_fastest_actions(env, state_dict_list)]

    @staticmethod
    def _select_from_scores(scores: Sequence[ActionScore]) -> int:

        if not scores:
            raise ValueError("No candidate new fastest VM action is available for policy selection.")
        selected = min(scores, key=lambda score: (score.vm_type_index, score.incremental_vm_cost, score.action))
        return int(selected.action)

    def select_action(self, env, state_dict_list: Optional[dict] = None) -> int:

        scores = self.score_candidate_actions(env, state_dict_list)
        return self._select_from_scores(scores)

    def select_action_with_diagnostics(self, env, state_dict_list: Optional[dict] = None):

        scores = self.score_candidate_actions(env, state_dict_list)
        action = self._select_from_scores(scores)
        selected_score = next(score for score in scores if score.action == action)
        diagnostics = {
            "feasible_new_fastest_action_num": int(len(scores)),
            "selected_meet_subdeadline": float(selected_score.meet_sub_deadline),
            "selected_incremental_vm_cost": float(selected_score.incremental_vm_cost),
            "selected_finish_time": float(selected_score.estimated_finish_time),
            "selected_task_sub_deadline": float(selected_score.task_sub_deadline),
            "selected_vm_cpu": float(selected_score.vm_cpu),
            "selected_vm_type_index": int(selected_score.vm_type_index),
            "selected_is_new_vm": 1.0,
        }
        return int(action), diagnostics


class FastestNewVMEvaluator:


    def __init__(self, env_test, policy, config: dict):
        self.env_test = env_test
        self.policy = policy
        self.config = config

    @staticmethod
    def _mean_or_zero(values) -> float:
        values = [float(value) for value in values]
        return float(sum(values) / len(values)) if values else 0.0

    def collect_one_episode(self, outer_iter: int, ep_num: int, mode: str = "test") -> EpisodeStats:

        if mode != "test":
            raise ValueError("FastestNewVMEvaluator is intended for deterministic test-mode evaluation.")

        dataset_group_idx, dataset_instance_idx = resolve_env_episode_indices(
            self.env_test,
            self.config,
            mode=mode,
            outer_iter=outer_iter,
            ep_num=ep_num,
        )
        episode_gamma = get_gamma_for_episode(self.env_test, self.config, mode, outer_iter, ep_num)
        self.env_test.set.gamma = episode_gamma
        state_dict = self.env_test.reset(seed=dataset_group_idx, ep_num=dataset_instance_idx, train_or_test=mode)

        total_reward = 0.0
        done = False
        decision_count = 0
        selected_vm_cpus = []
        selected_new_vm_flags = []
        selected_safe_flags = []

        while not done:
            action, diagnostics = self.policy.select_action_with_diagnostics(self.env_test, state_dict)
            next_state_dict, reward, done, _ = self.env_test.step({"0": action})
            total_reward += float(reward)
            decision_count += 1
            selected_vm_cpus.append(float(diagnostics["selected_vm_cpu"]))
            selected_new_vm_flags.append(float(diagnostics["selected_is_new_vm"]))
            selected_safe_flags.append(float(diagnostics["selected_meet_subdeadline"]))
            state_dict = next_state_dict

        episode_info = self.env_test.episode_info
        workflow_metrics = aggregate_workflow_lateness_metrics(
            workflow_normalized_lateness=episode_info["workflow_normalized_lateness"],
            workflow_soft_clipped_normalized_lateness=episode_info["workflow_soft_clipped_normalized_lateness"],
        )

        return EpisodeStats(
            total_reward=float(total_reward),
            total_vm_cost=float(episode_info["VM_cost"]),
            total_deadline_penalty=float(episode_info["SLA_penalty"]),
            miss_deadline_num=int(episode_info["missDeadlineNum"]),
            vm_exec_hour=float(episode_info["VM_execHour"]),
            vm_tot_hour=float(episode_info["VM_totHour"]),
            gamma_value=to_serializable(episode_gamma),
            wf_count=workflow_metrics["wf_count"],
            deadline_met_wf_count=workflow_metrics["deadline_met_wf_count"],
            deadline_missed_wf_count=workflow_metrics["deadline_missed_wf_count"],
            norm_lateness_sum_all_wf=workflow_metrics["norm_lateness_sum_all_wf"],
            norm_lateness_sum_missed_wf=workflow_metrics["norm_lateness_sum_missed_wf"],
            soft_clip_norm_lateness_sum_all_wf=workflow_metrics["soft_clip_norm_lateness_sum_all_wf"],
            soft_clip_norm_lateness_sum_missed_wf=workflow_metrics["soft_clip_norm_lateness_sum_missed_wf"],
            decision_count=int(decision_count),
            selected_vm_cpu_mean=self._mean_or_zero(selected_vm_cpus),
            selected_new_vm_rate=self._mean_or_zero(selected_new_vm_flags),
            selected_action_meet_subdeadline_rate=self._mean_or_zero(selected_safe_flags),
        )

    def collect_episodes(self, n_episodes: int, outer_iter: int = 0) -> List[EpisodeStats]:

        if int(n_episodes) <= 0:
            raise ValueError(f"Evaluation requires at least one episode, but got {n_episodes}.")
        return [self.collect_one_episode(outer_iter=outer_iter, ep_num=ep_num, mode="test") for ep_num in range(int(n_episodes))]

    @staticmethod
    def _aggregate_eval_stats(episodes: List[EpisodeStats]) -> EvalStats:
        if not episodes:
            raise ValueError("At least one completed episode is required for aggregation.")

        wf_count = sum(ep.wf_count for ep in episodes)
        deadline_met_wf_count = sum(ep.deadline_met_wf_count for ep in episodes)
        deadline_missed_wf_count = sum(ep.deadline_missed_wf_count for ep in episodes)
        norm_lateness_sum_all_wf = sum(ep.norm_lateness_sum_all_wf for ep in episodes)
        norm_lateness_sum_missed_wf = sum(ep.norm_lateness_sum_missed_wf for ep in episodes)
        soft_clip_norm_lateness_sum_all_wf = sum(ep.soft_clip_norm_lateness_sum_all_wf for ep in episodes)
        soft_clip_norm_lateness_sum_missed_wf = sum(ep.soft_clip_norm_lateness_sum_missed_wf for ep in episodes)
        vm_cost_total = sum(ep.total_vm_cost for ep in episodes)
        decision_count = sum(ep.decision_count for ep in episodes)


        selected_vm_cpu_sum = sum(ep.selected_vm_cpu_mean * ep.decision_count for ep in episodes)
        selected_new_vm_sum = sum(ep.selected_new_vm_rate * ep.decision_count for ep in episodes)
        selected_safe_sum = sum(ep.selected_action_meet_subdeadline_rate * ep.decision_count for ep in episodes)

        return EvalStats(
            avg_return=float(sum(ep.total_reward for ep in episodes) / len(episodes)),
            avg_vm_exec_hour=float(sum(ep.vm_exec_hour for ep in episodes) / len(episodes)),
            avg_vm_tot_hour=float(sum(ep.vm_tot_hour for ep in episodes) / len(episodes)),
            avg_vm_cost=float(sum(ep.total_vm_cost for ep in episodes) / len(episodes)),
            avg_deadline_penalty=float(sum(ep.total_deadline_penalty for ep in episodes) / len(episodes)),
            avg_miss_deadline_num=float(sum(ep.miss_deadline_num for ep in episodes) / len(episodes)),
            gamma_per_instance=[ep.gamma_value for ep in episodes],
            wf_count=int(wf_count),
            deadline_met_wf_count=int(deadline_met_wf_count),
            deadline_missed_wf_count=int(deadline_missed_wf_count),
            deadline_met_wf_rate=float(deadline_met_wf_count / wf_count) if wf_count > 0 else 0.0,
            deadline_missed_wf_rate=float(deadline_missed_wf_count / wf_count) if wf_count > 0 else 0.0,
            norm_lateness_mean_all_wf=float(norm_lateness_sum_all_wf / wf_count) if wf_count > 0 else 0.0,
            norm_lateness_mean_missed_wf=(
                float(norm_lateness_sum_missed_wf / deadline_missed_wf_count)
                if deadline_missed_wf_count > 0 else 0.0
            ),
            soft_clip_norm_lateness_mean_all_wf=(
                float(soft_clip_norm_lateness_sum_all_wf / wf_count) if wf_count > 0 else 0.0
            ),
            soft_clip_norm_lateness_mean_missed_wf=(
                float(soft_clip_norm_lateness_sum_missed_wf / deadline_missed_wf_count)
                if deadline_missed_wf_count > 0 else 0.0
            ),
            vm_cost_total=float(vm_cost_total),
            vm_cost_mean_per_wf=float(vm_cost_total / wf_count) if wf_count > 0 else 0.0,
            decision_count=int(decision_count),
            selected_vm_cpu_mean=float(selected_vm_cpu_sum / decision_count) if decision_count > 0 else 0.0,
            selected_new_vm_rate=float(selected_new_vm_sum / decision_count) if decision_count > 0 else 0.0,
            selected_action_meet_subdeadline_rate=(
                float(selected_safe_sum / decision_count) if decision_count > 0 else 0.0
            ),
        )

    def evaluate(self, n_episodes: int, outer_iter: int = 0) -> EvalStats:

        episodes = self.collect_episodes(n_episodes=n_episodes, outer_iter=outer_iter)
        return self._aggregate_eval_stats(episodes)
