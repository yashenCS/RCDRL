from typing import List
import torch
from algorithms.rcdrl.core.types import Episode, Transition
from algorithms.rcdrl.core.utils import (
    filtered_feasible_actions,
    get_gamma_for_episode,
    parse_env_state,
    resolve_env_episode_indices,
    to_serializable,
)


class EpisodeCollector:
    def __init__(self, device, env, policy, config: dict):
        self.device = device
        self.env = env
        self.policy = policy
        self.config = config

    @staticmethod
    def _compute_task_subdeadline_target_for_action(
        env, action: int, eps_den: float = 1e-08
    ) -> float:
        current_wf = getattr(env, "nextWrf", None)
        current_task = getattr(env, "nextTask", None)
        decision_time = getattr(env, "nextTimeStep", None)
        if current_wf is None or current_task is None or decision_time is None:
            raise RuntimeError(
                "Cannot compute sub-deadline target because env.nextWrf, env.nextTask, or env.nextTimeStep is None."
            )
        if not hasattr(env, "appSubDeadline"):
            raise RuntimeError(
                "Cannot compute sub-deadline target because env.appSubDeadline is missing."
            )
        if current_wf not in env.appSubDeadline:
            raise KeyError("Missing sub-deadline dictionary for the current workflow.")
        if current_task not in env.appSubDeadline[current_wf]:
            raise KeyError(
                f"Missing sub-deadline for task {current_task}. Available task keys: {list(env.appSubDeadline[current_wf].keys())}"
            )
        action = int(action)
        decision_time = float(decision_time)
        sub_deadline = float(env.appSubDeadline[current_wf][current_task])
        task_process_time = float(current_wf.get_taskProcessTime(current_task))
        if action < len(env.vm_queues):
            selected_vm = env.vm_queues[action]
            selected_cpu = float(env.vm_queues_cpu[action])
            estimated_start_time = max(decision_time, float(selected_vm.vmLatestTime()))
        else:
            diff = action - len(env.vm_queues)
            if diff < 0:
                raise ValueError(
                    f"Invalid VM action index {action} for {len(env.vm_queues)} active VMs."
                )
            vmtype = diff % int(env.VMtypeNum)
            selected_cpu = float(env.set.dataset.vmVCPU[vmtype])
            estimated_start_time = decision_time
        task_execution_time = task_process_time / max(selected_cpu, eps_den)
        estimated_finish_time = estimated_start_time + task_execution_time
        lateness = max(0.0, estimated_finish_time - sub_deadline)
        normalized_lateness = lateness / max(sub_deadline - decision_time, eps_den)
        soft_clipped_lateness = normalized_lateness / (1.0 + normalized_lateness)
        return float(soft_clipped_lateness)

    @classmethod
    def _compute_candidate_task_subdeadline_targets(
        cls, env, action_num: int, remove_vm=None, eps_den: float = 1e-08
    ):
        action_num = int(action_num)
        if action_num <= 0:
            raise ValueError(f"action_num must be positive, but got {action_num}.")
        targets = [
            cls._compute_task_subdeadline_target_for_action(
                env, action, eps_den=eps_den
            )
            for action in range(action_num)
        ]
        feasible_actions = filtered_feasible_actions(action_num, remove_vm)
        return (targets, feasible_actions)

    @staticmethod
    def _aggregate_workflow_lateness_metrics(
        workflow_normalized_lateness: dict,
        workflow_soft_clipped_normalized_lateness: dict,
    ) -> dict:
        normalized_values = [
            float(value) for value in workflow_normalized_lateness.values()
        ]
        soft_values = [
            float(value) for value in workflow_soft_clipped_normalized_lateness.values()
        ]
        if len(normalized_values) != len(soft_values):
            raise RuntimeError(
                f"workflow_normalized_lateness and workflow_soft_clipped_normalized_lateness have different sizes: {len(normalized_values)} vs {len(soft_values)}."
            )
        wf_count = len(normalized_values)
        deadline_missed_wf_count = sum(
            (1 for value in normalized_values if value > 0.0)
        )
        deadline_met_wf_count = wf_count - deadline_missed_wf_count
        missed_keys = [
            key
            for key, value in workflow_normalized_lateness.items()
            if float(value) > 0.0
        ]
        return {
            "wf_count": int(wf_count),
            "deadline_met_wf_count": int(deadline_met_wf_count),
            "deadline_missed_wf_count": int(deadline_missed_wf_count),
            "norm_lateness_sum_all_wf": float(sum(normalized_values)),
            "norm_lateness_sum_missed_wf": float(
                sum((float(workflow_normalized_lateness[key]) for key in missed_keys))
            ),
            "soft_clip_norm_lateness_sum_all_wf": float(sum(soft_values)),
            "soft_clip_norm_lateness_sum_missed_wf": float(
                sum(
                    (
                        float(workflow_soft_clipped_normalized_lateness[key])
                        for key in missed_keys
                    )
                )
            ),
        }

    def collect_one_episode(
        self,
        model,
        outer_iter: int,
        ep_num: int,
        mode: str = "train",
        explore: bool = True,
    ) -> Episode:
        group_idx, instance_idx = resolve_env_episode_indices(
            self.env, self.config, mode, outer_iter, ep_num
        )
        episode_gamma = get_gamma_for_episode(
            self.env, self.config, mode, outer_iter, ep_num
        )
        self.env.set.gamma = episode_gamma
        state_dict = self.env.reset(
            seed=group_idx, ep_num=instance_idx, train_or_test=mode
        )
        state = parse_env_state(state_dict)
        transitions: List[Transition] = []
        total_reward = 0.0
        done = False
        modules = (model.encoder, model.deadline_head, model.vm_critic)
        previous_modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            while not done:
                with torch.no_grad():
                    scores = model.forward_critic(self.device, state)
                risks = scores["subdeadline_risk_all"].detach().cpu().numpy()
                costs = scores["q_vm_critic_all"].detach().cpu().numpy()
                action = self.policy.select_behavior_action(
                    risks, costs, remove_vm=state.remove_vm, explore=explore
                )
                targets, feasible = self._compute_candidate_task_subdeadline_targets(
                    self.env, action_num=len(risks), remove_vm=state.remove_vm
                )
                previous_cost = float(self.env.VMcost)
                next_state_dict, reward, done, _ = self.env.step({"0": action})
                current_cost = (
                    float(self.env.episode_info["VM_cost"])
                    if done
                    else float(self.env.VMcost)
                )
                step_vm_cost = current_cost - previous_cost
                if step_vm_cost < -1e-08:
                    raise ValueError(f"Negative step_vm_cost detected: {step_vm_cost}")
                next_state = None if done else parse_env_state(next_state_dict)
                transitions.append(
                    Transition(
                        state=state,
                        action=int(action),
                        reward=float(reward),
                        step_vm_cost=step_vm_cost,
                        next_state=next_state,
                        done=bool(done),
                        y_deadline_all=list(targets),
                        y_deadline_feasible_actions=list(feasible),
                    )
                )
                total_reward += float(reward)
                state = next_state
        finally:
            for module, was_training in zip(modules, previous_modes):
                module.train(was_training)
        info = self.env.episode_info
        workflow_metrics = self._aggregate_workflow_lateness_metrics(
            info["workflow_normalized_lateness"],
            info["workflow_soft_clipped_normalized_lateness"],
        )
        return Episode(
            transitions=transitions,
            total_reward=total_reward,
            total_vm_cost=float(info["VM_cost"]),
            total_deadline_penalty=float(info["SLA_penalty"]),
            miss_deadline_num=int(info["missDeadlineNum"]),
            deadline_violated=int(info["missDeadlineNum"] > 0),
            vm_exec_hour=float(info["VM_execHour"]),
            vm_tot_hour=float(info["VM_totHour"]),
            gamma_value=to_serializable(episode_gamma),
            **workflow_metrics,
        )

    def collect_episodes(
        self,
        model,
        outer_iter: int,
        n_episodes: int,
        mode: str = "train",
        explore: bool = True,
    ):
        return [
            self.collect_one_episode(
                model, outer_iter=outer_iter, ep_num=ep_num, mode=mode, explore=explore
            )
            for ep_num in range(n_episodes)
        ]
