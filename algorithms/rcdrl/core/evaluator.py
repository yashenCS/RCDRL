import torch
from algorithms.rcdrl.core.collector import EpisodeCollector
from algorithms.rcdrl.core.types import EvalStats


class RCDRLEvaluator:
    def __init__(self, device, env_test, policy, config: dict):
        self.collector = EpisodeCollector(device, env_test, policy, config=config)

    @staticmethod
    def _workflow_metric_row_from_episodes(episodes) -> dict:
        wf_count = sum((ep.wf_count for ep in episodes))
        deadline_met_wf_count = sum((ep.deadline_met_wf_count for ep in episodes))
        deadline_missed_wf_count = sum((ep.deadline_missed_wf_count for ep in episodes))
        norm_lateness_sum_all_wf = sum((ep.norm_lateness_sum_all_wf for ep in episodes))
        norm_lateness_sum_missed_wf = sum(
            (ep.norm_lateness_sum_missed_wf for ep in episodes)
        )
        soft_clip_norm_lateness_sum_all_wf = sum(
            (ep.soft_clip_norm_lateness_sum_all_wf for ep in episodes)
        )
        soft_clip_norm_lateness_sum_missed_wf = sum(
            (ep.soft_clip_norm_lateness_sum_missed_wf for ep in episodes)
        )
        vm_cost_total = sum((ep.total_vm_cost for ep in episodes))
        return {
            "wf_count": int(wf_count),
            "deadline_met_wf_count": int(deadline_met_wf_count),
            "deadline_missed_wf_count": int(deadline_missed_wf_count),
            "deadline_met_wf_rate": float(deadline_met_wf_count / wf_count)
            if wf_count > 0
            else 0.0,
            "deadline_missed_wf_rate": float(deadline_missed_wf_count / wf_count)
            if wf_count > 0
            else 0.0,
            "norm_lateness_mean_all_wf": float(norm_lateness_sum_all_wf / wf_count)
            if wf_count > 0
            else 0.0,
            "norm_lateness_mean_missed_wf": float(
                norm_lateness_sum_missed_wf / deadline_missed_wf_count
            )
            if deadline_missed_wf_count > 0
            else 0.0,
            "soft_clip_norm_lateness_mean_all_wf": float(
                soft_clip_norm_lateness_sum_all_wf / wf_count
            )
            if wf_count > 0
            else 0.0,
            "soft_clip_norm_lateness_mean_missed_wf": float(
                soft_clip_norm_lateness_sum_missed_wf / deadline_missed_wf_count
            )
            if deadline_missed_wf_count > 0
            else 0.0,
            "vm_cost_total": float(vm_cost_total),
            "vm_cost_mean_per_wf": float(vm_cost_total / wf_count)
            if wf_count > 0
            else 0.0,
        }

    @torch.no_grad()
    def evaluate(self, model, outer_iter: int, n_episodes: int) -> EvalStats:
        if int(n_episodes) <= 0:
            raise ValueError(
                f"Evaluation requires at least one episode, but got {n_episodes}."
            )
        episodes = self.collector.collect_episodes(
            model=model,
            outer_iter=outer_iter,
            n_episodes=n_episodes,
            mode="test",
            explore=False,
        )
        avg_return = sum((ep.total_reward for ep in episodes)) / len(episodes)
        avg_vm_exec_hour = sum((ep.vm_exec_hour for ep in episodes)) / len(episodes)
        avg_vm_tot_hour = sum((ep.vm_tot_hour for ep in episodes)) / len(episodes)
        avg_vm_cost = sum((ep.total_vm_cost for ep in episodes)) / len(episodes)
        avg_deadline_penalty = sum(
            (ep.total_deadline_penalty for ep in episodes)
        ) / len(episodes)
        avg_miss_deadline_num = sum((ep.miss_deadline_num for ep in episodes)) / len(
            episodes
        )
        gamma_per_instance = [ep.gamma_value for ep in episodes]
        workflow_metrics = self._workflow_metric_row_from_episodes(episodes)
        return EvalStats(
            avg_return=float(avg_return),
            avg_vm_exec_hour=float(avg_vm_exec_hour),
            avg_vm_tot_hour=float(avg_vm_tot_hour),
            avg_vm_cost=float(avg_vm_cost),
            avg_deadline_penalty=float(avg_deadline_penalty),
            avg_miss_deadline_num=float(avg_miss_deadline_num),
            gamma_per_instance=gamma_per_instance,
            wf_count=workflow_metrics["wf_count"],
            deadline_met_wf_count=workflow_metrics["deadline_met_wf_count"],
            deadline_missed_wf_count=workflow_metrics["deadline_missed_wf_count"],
            deadline_met_wf_rate=workflow_metrics["deadline_met_wf_rate"],
            deadline_missed_wf_rate=workflow_metrics["deadline_missed_wf_rate"],
            norm_lateness_mean_all_wf=workflow_metrics["norm_lateness_mean_all_wf"],
            norm_lateness_mean_missed_wf=workflow_metrics[
                "norm_lateness_mean_missed_wf"
            ],
            soft_clip_norm_lateness_mean_all_wf=workflow_metrics[
                "soft_clip_norm_lateness_mean_all_wf"
            ],
            soft_clip_norm_lateness_mean_missed_wf=workflow_metrics[
                "soft_clip_norm_lateness_mean_missed_wf"
            ],
            vm_cost_total=workflow_metrics["vm_cost_total"],
            vm_cost_mean_per_wf=workflow_metrics["vm_cost_mean_per_wf"],
        )
