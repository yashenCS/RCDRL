import csv
import json
import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from algorithms.rcdrl.core.reference import FastestNewVMEvaluator
from algorithms.rcdrl.core.reference import FastestNewVMPolicy
from algorithms.rcdrl.core.collector import EpisodeCollector
from algorithms.rcdrl.core.evaluator import RCDRLEvaluator
from algorithms.rcdrl.core.policy import SafeFirstPolicy
from algorithms.rcdrl.core.selection import choose_better_candidate, constraint_metrics
from algorithms.rcdrl.core.utils import (
    ensure_dir,
    freeze_module,
    hard_update,
    soft_update,
    to_serializable,
    unfreeze_module,
)

RCDRL_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)


class RCDRLTrainer:
    def __init__(
        self,
        device,
        config,
        env,
        env_test,
        model,
        rollout_buffer,
        train_Set_setting,
        test_Set_setting,
    ):
        self.device = device
        self.config = config
        self.env = env
        self.env_test = env_test
        self.model = model.to(device)
        self.rollout_buffer = rollout_buffer
        yaml_cfg = config.config["yaml-config"]
        self.env_cfg = yaml_cfg["env"]
        self.training_cfg = yaml_cfg["training"]
        self.optim_cfg = yaml_cfg["optim"]
        self.runtime_cfg = yaml_cfg.get("runtime", {})
        self.constraint_cfg = yaml_cfg.get("constraint", {})
        self.evaluation_cfg = yaml_cfg.get("evaluation", {})
        self.outer_iters = int(self.training_cfg["outer_iters"])
        self.rollout_episodes_per_iter = int(
            self.training_cfg["rollout_episodes_per_iter"]
        )
        self.gamma_vm_td = float(self.training_cfg["gamma_vm_td"])
        self.stage1_updates_per_iter = int(self.training_cfg["stage1_updates_per_iter"])
        self.stage2_updates_per_iter = int(self.training_cfg["stage2_updates_per_iter"])
        self.batch_size_stage1 = int(self.training_cfg["batch_size_stage1"])
        self.batch_size_stage2 = int(self.training_cfg["batch_size_stage2"])
        self.vm_target_tau = float(self.training_cfg["vm_target_tau"])
        self.grad_clip = (
            None
            if self.training_cfg["grad_clip"] is None
            else float(self.training_cfg["grad_clip"])
        )
        self.eval_freq = int(self.training_cfg["eval_freq"])
        self.save_freq = int(self.training_cfg["save_freq"])
        self.valid_ep_num = int(self.env_cfg["validNum"])
        self.relative_deadline_gap = float(
            self.constraint_cfg.get("relative_deadline_gap", 0.1)
        )
        self.validation_seed = int(self.evaluation_cfg.get("validation_seed", 41))
        self.policy = SafeFirstPolicy(
            delta=float(self.training_cfg["delta"]),
            epsilon=float(self.training_cfg["epsilon"]),
            fallback_top_ratio=float(self.training_cfg["fallback_top_ratio"]),
        )
        self.collector = EpisodeCollector(device, env, self.policy, config=yaml_cfg)
        self.evaluator = RCDRLEvaluator(device, env_test, self.policy, config=yaml_cfg)
        self.optim_stage1 = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.deadline_head.parameters()),
            lr=float(self.optim_cfg["lr_stage1"]),
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        self.optim_stage2 = torch.optim.Adam(
            self.model.vm_critic.parameters(),
            lr=float(self.optim_cfg["lr_stage2"]),
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
        )
        self.log_enabled = bool(self.runtime_cfg.get("log", True))
        self.save_mode_dir = None
        self.saved_models_dir = None
        self.train_perf_csv_path = None
        self.validation_perf_csv_path = None
        self.validation_reference = None
        self.best_validation_candidate = None

    def _init_logging(self) -> None:
        if not self.log_enabled:
            return
        now = datetime.now().strftime("%Y%m%d%H%M%S%f")
        seed = int(self.env_cfg.get("seed", 0))
        self.save_mode_dir = os.path.join(
            RCDRL_LOG_DIR, self.env.name, f"RCDRL_seed{seed}_{now}_{os.getpid()}"
        )
        self.saved_models_dir = os.path.join(self.save_mode_dir, "saved_models")
        train_perf_dir = os.path.join(self.save_mode_dir, "train_performance")
        validation_perf_dir = os.path.join(self.save_mode_dir, "validation_performance")
        ensure_dir(self.saved_models_dir)
        ensure_dir(train_perf_dir)
        ensure_dir(validation_perf_dir)
        with open(
            os.path.join(self.save_mode_dir, "profile.yaml"), "w", encoding="utf-8"
        ) as f:
            import yaml

            yaml.dump(self.config.config["yaml-config"], f)
        self.train_perf_csv_path = os.path.join(train_perf_dir, "training_record.csv")
        self.validation_perf_csv_path = os.path.join(
            validation_perf_dir, "validation_record.csv"
        )

    def _append_csv(self, csv_path: str, row: dict) -> None:
        if not self.log_enabled or csv_path is None:
            return
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_json(self, path: str, payload: dict) -> None:
        if not self.log_enabled or path is None:
            return
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_serializable(payload), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)

    def save_checkpoint(self, outer_iter: int):
        if not self.log_enabled:
            return None
        model_idx = self._model_num_from_outer_iter(outer_iter)
        ckpt_path = os.path.join(self.saved_models_dir, f"ep_{model_idx}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        return ckpt_path

    @staticmethod
    def _model_num_from_outer_iter(outer_iter: int) -> int:
        return int(outer_iter) + 1

    def _csv_context_row(self, outer_iter: int) -> dict:
        return {
            "model_num": self._model_num_from_outer_iter(outer_iter),
            "wf_size": self.env_cfg.get("wf_size", ""),
        }

    def _init_validation_selection(self) -> None:
        reference_evaluator = FastestNewVMEvaluator(
            env_test=self.env_test,
            policy=FastestNewVMPolicy(),
            config=self.config.config["yaml-config"],
        )
        stats = reference_evaluator.evaluate(n_episodes=self.valid_ep_num, outer_iter=0)
        self.validation_reference = {
            "split": "validation_reference",
            "fixed_seed": self.validation_seed,
            "episodes": self.valid_ep_num,
            "workflow_count": stats.wf_count,
            "deadline_met_count": stats.deadline_met_wf_count,
            "deadline_missed_count": stats.deadline_missed_wf_count,
            "success_rate": stats.deadline_met_wf_rate,
            "miss_rate": stats.deadline_missed_wf_rate,
            "mean_vm_cost_per_episode": stats.avg_vm_cost,
            "total_vm_cost": stats.vm_cost_total,
            "mean_environment_reward": stats.avg_return,
        }
        protocol = constraint_metrics(
            success_rate=stats.deadline_met_wf_rate,
            reference_success_rate=stats.deadline_met_wf_rate,
            relative_deadline_gap=self.relative_deadline_gap,
        )
        self._write_json(
            os.path.join(self.save_mode_dir, "validation_reference.json")
            if self.save_mode_dir
            else None,
            self.validation_reference,
        )
        self._write_json(
            os.path.join(self.save_mode_dir, "constraint_definition.json")
            if self.save_mode_dir
            else None,
            {
                "relative_deadline_gap": self.relative_deadline_gap,
                "validation_reference_success": stats.deadline_met_wf_rate,
                "minimum_success_rate": protocol["minimum_success_rate"],
                "maximum_miss_rate": protocol["maximum_miss_rate"],
            },
        )
        print(
            f"Validation protocol: seed={self.validation_seed}, FASTEST-NEW-VM success={stats.deadline_met_wf_rate:.4f}, minimum success={protocol['minimum_success_rate']:.4f}",
            flush=True,
        )

    def _consider_validation_checkpoint(
        self, outer_iter: int, eval_stats, checkpoint_path: str
    ) -> dict:
        if checkpoint_path is None:
            raise RuntimeError(
                "Validation checkpoint selection requires experiment logging to be enabled."
            )
        protocol = constraint_metrics(
            success_rate=eval_stats.deadline_met_wf_rate,
            reference_success_rate=self.validation_reference["success_rate"],
            relative_deadline_gap=self.relative_deadline_gap,
        )
        candidate = {
            "iteration": int(outer_iter) + 1,
            "model_num": self._model_num_from_outer_iter(outer_iter),
            "checkpoint": checkpoint_path,
            "success_rate": eval_stats.deadline_met_wf_rate,
            "miss_rate": eval_stats.deadline_missed_wf_rate,
            "mean_vm_cost_per_episode": eval_stats.avg_vm_cost,
            "total_vm_cost": eval_stats.vm_cost_total,
            **protocol,
        }
        self.best_validation_candidate = choose_better_candidate(
            self.best_validation_candidate, candidate
        )
        return protocol

    def _finalize_checkpoint_selection(self) -> None:
        if not self.log_enabled:
            return
        if self.best_validation_candidate is None:
            raise RuntimeError(
                "Training completed without any validation checkpoint candidate."
            )
        selected_checkpoint = self.best_validation_candidate["checkpoint"]
        if not os.path.isfile(selected_checkpoint):
            raise FileNotFoundError(
                f"Selected checkpoint does not exist: {selected_checkpoint}"
            )
        self._write_json(
            os.path.join(self.save_mode_dir, "selection.json"),
            {
                "status": "complete",
                "algorithm": "RCDRL",
                "training_seed": int(self.env_cfg["seed"]),
                "trained_workflow_size": str(self.env_cfg["wf_size"]),
                "completed_outer_iters": self.outer_iters,
                "validation_seed": self.validation_seed,
                "relative_deadline_gap": self.relative_deadline_gap,
                "selection_rule": "lowest validation VM cost among feasible checkpoints; otherwise lowest success shortfall, then VM cost; ties prefer the earlier checkpoint",
                "selected_model_num": int(self.best_validation_candidate["model_num"]),
                "selected_checkpoint": selected_checkpoint,
                "validation_reference": self.validation_reference,
                "validation": self.best_validation_candidate,
            },
        )

    def _deadline_loss_for_transition_all_feasible(
        self, outputs: dict, transition
    ) -> torch.Tensor:
        feasible = torch.as_tensor(
            transition.y_deadline_feasible_actions, dtype=torch.long, device=self.device
        )
        targets = torch.as_tensor(
            transition.y_deadline_all, dtype=torch.float32, device=self.device
        )
        logits = outputs["subdeadline_logit_all"]
        if targets.ndim != 1 or targets.shape != logits.shape:
            raise ValueError("Deadline targets must match the candidate-action logits.")
        if feasible.ndim != 1 or feasible.numel() == 0:
            raise ValueError(
                "Deadline supervision requires a non-empty feasible-action list."
            )
        if int(feasible.min()) < 0 or int(feasible.max()) >= targets.numel():
            raise ValueError(
                "Feasible-action index is outside the deadline-target array."
            )
        return F.binary_cross_entropy_with_logits(
            logits.index_select(0, feasible), targets.index_select(0, feasible)
        )

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

    @staticmethod
    def _workflow_metric_row_from_eval_stats(eval_stats) -> dict:
        return {
            "wf_count": eval_stats.wf_count,
            "deadline_met_wf_count": eval_stats.deadline_met_wf_count,
            "deadline_missed_wf_count": eval_stats.deadline_missed_wf_count,
            "deadline_met_wf_rate": eval_stats.deadline_met_wf_rate,
            "deadline_missed_wf_rate": eval_stats.deadline_missed_wf_rate,
            "norm_lateness_mean_all_wf": eval_stats.norm_lateness_mean_all_wf,
            "norm_lateness_mean_missed_wf": eval_stats.norm_lateness_mean_missed_wf,
            "soft_clip_norm_lateness_mean_all_wf": eval_stats.soft_clip_norm_lateness_mean_all_wf,
            "soft_clip_norm_lateness_mean_missed_wf": eval_stats.soft_clip_norm_lateness_mean_missed_wf,
            "vm_cost_total": eval_stats.vm_cost_total,
            "vm_cost_mean_per_wf": eval_stats.vm_cost_mean_per_wf,
        }

    @staticmethod
    def _episode_summary_row(
        episodes, outer_iter: int, buffer_size: int, stage_stats: dict
    ) -> dict:
        if not episodes:
            raise ValueError("At least one completed episode is required.")
        count = len(episodes)
        return {
            "outer_iter": outer_iter + 1,
            "buffer_size": buffer_size,
            "rewards": sum((ep.total_reward for ep in episodes)) / count,
            "VM_execHour": sum((ep.vm_exec_hour for ep in episodes)) / count,
            "VM_totHour": sum((ep.vm_tot_hour for ep in episodes)) / count,
            "VM_cost": sum((ep.total_vm_cost for ep in episodes)) / count,
            "deadline_penalty": sum((ep.total_deadline_penalty for ep in episodes))
            / count,
            "missDeadlineNum": sum((ep.miss_deadline_num for ep in episodes)) / count,
            "gamma_per_instance": json.dumps(
                [to_serializable(ep.gamma_value) for ep in episodes]
            ),
            **RCDRLTrainer._workflow_metric_row_from_episodes(episodes),
            **stage_stats,
        }

    def _clip_if_needed(self, params) -> None:
        if self.grad_clip is not None:
            clip_grad_norm_(params, self.grad_clip)

    def train(self) -> None:
        self._init_logging()
        self._init_validation_selection()
        hard_update(self.model.vm_critic_target, self.model.vm_critic)
        for outer_iter in range(self.outer_iters):
            self.rollout_buffer.clear()
            episodes = self.collector.collect_episodes(
                self.model,
                outer_iter=outer_iter,
                n_episodes=self.rollout_episodes_per_iter,
                mode="train",
                explore=True,
            )
            for episode in episodes:
                self.rollout_buffer.add_episode(episode)
            started = time.perf_counter()
            stage_stats = self.run_stage1()
            stage1_time = time.perf_counter() - started
            started = time.perf_counter()
            stage_stats.update(self.run_stage2())
            stage2_time = time.perf_counter() - started
            row = self._episode_summary_row(
                episodes, outer_iter, len(self.rollout_buffer), stage_stats
            )
            row.update(self._csv_context_row(outer_iter))
            self._append_csv(self.train_perf_csv_path, row)
            print(
                f"iter={outer_iter + 1} train_cost={row['VM_cost']:.4f} success={row['deadline_met_wf_rate']:.4f} stage1_loss={stage_stats['stage1_loss']:.6f} stage2_td_loss={stage_stats['stage2_vm_td_loss']:.6f} stage1_time={stage1_time:.3f}s stage2_time={stage2_time:.3f}s",
                flush=True,
            )
            iteration = outer_iter + 1
            should_validate = (
                iteration % self.eval_freq == 0 or iteration == self.outer_iters
            )
            should_save = (
                iteration % self.save_freq == 0 or iteration == self.outer_iters
            )
            checkpoint = self.save_checkpoint(outer_iter) if should_save else None
            if should_validate:
                if checkpoint is None:
                    checkpoint = self.save_checkpoint(outer_iter)
                stats = self.evaluator.evaluate(
                    self.model, outer_iter=outer_iter, n_episodes=self.valid_ep_num
                )
                if self.log_enabled:
                    protocol = self._consider_validation_checkpoint(
                        outer_iter, stats, checkpoint
                    )
                else:
                    protocol = constraint_metrics(
                        success_rate=stats.deadline_met_wf_rate,
                        reference_success_rate=self.validation_reference[
                            "success_rate"
                        ],
                        relative_deadline_gap=self.relative_deadline_gap,
                    )
                eval_row = {
                    "outer_iter": iteration,
                    "split": "validation",
                    "fixed_seed": self.validation_seed,
                    "rewards": stats.avg_return,
                    "VM_execHour": stats.avg_vm_exec_hour,
                    "VM_totHour": stats.avg_vm_tot_hour,
                    "VM_cost": stats.avg_vm_cost,
                    "deadline_penalty": stats.avg_deadline_penalty,
                    "missDeadlineNum": stats.avg_miss_deadline_num,
                    "gamma_per_instance": json.dumps(stats.gamma_per_instance),
                    **protocol,
                    **self._csv_context_row(outer_iter),
                    **self._workflow_metric_row_from_eval_stats(stats),
                }
                self._append_csv(self.validation_perf_csv_path, eval_row)
                print(
                    f"validation iter={iteration} cost={stats.avg_vm_cost:.4f} success={stats.deadline_met_wf_rate:.4f} feasible={protocol['feasible']}",
                    flush=True,
                )
        self._finalize_checkpoint_selection()

    def eval(
        self, policy_path: str, outer_iter: int = 0, n_episodes: int = None
    ) -> dict:
        state_dict = torch.load(policy_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        episode_num = self.valid_ep_num if n_episodes is None else int(n_episodes)
        eval_stats = self.evaluator.evaluate(
            self.model, outer_iter=outer_iter, n_episodes=episode_num
        )
        row = {
            "rewards": eval_stats.avg_return,
            "VM_execHour": eval_stats.avg_vm_exec_hour,
            "VM_totHour": eval_stats.avg_vm_tot_hour,
            "VM_cost": eval_stats.avg_vm_cost,
            "deadline_penalty": eval_stats.avg_deadline_penalty,
            "missDeadlineNum": eval_stats.avg_miss_deadline_num,
            "gamma_per_instance": json.dumps(eval_stats.gamma_per_instance),
        }
        row.update(
            {
                "model_num": int(
                    self.env_cfg.get(
                        "model_num", self._model_num_from_outer_iter(outer_iter)
                    )
                ),
                "wf_size": self.env_cfg.get("wf_size", ""),
            }
        )
        row.update(self._workflow_metric_row_from_eval_stats(eval_stats))
        return row

    def run_stage1(self) -> dict:
        if not len(self.rollout_buffer):
            return {"stage1_loss": 0.0}
        unfreeze_module(self.model.encoder)
        unfreeze_module(self.model.deadline_head)
        freeze_module(self.model.vm_critic)
        freeze_module(self.model.vm_critic_target)
        params = list(self.model.encoder.parameters()) + list(
            self.model.deadline_head.parameters()
        )
        total_loss = 0.0
        for _ in range(self.stage1_updates_per_iter):
            batch = self.rollout_buffer.sample(self.batch_size_stage1)
            self.optim_stage1.zero_grad()
            loss = sum(
                (
                    self._deadline_loss_for_transition_all_feasible(
                        self.model.forward_stage1(self.device, tr.state), tr
                    )
                    for tr in batch
                )
            ) / len(batch)
            loss.backward()
            self._clip_if_needed(params)
            self.optim_stage1.step()
            total_loss += float(loss.item())
        return {"stage1_loss": total_loss / self.stage1_updates_per_iter}

    def run_stage2(self) -> dict:
        if not len(self.rollout_buffer):
            return {"stage2_vm_td_loss": 0.0}
        freeze_module(self.model.encoder)
        freeze_module(self.model.deadline_head)
        self.model.encoder.zero_grad(set_to_none=True)
        self.model.deadline_head.zero_grad(set_to_none=True)
        unfreeze_module(self.model.vm_critic)
        freeze_module(self.model.vm_critic_target)
        params = list(self.model.vm_critic.parameters())
        total_loss = 0.0
        for _ in range(self.stage2_updates_per_iter):
            batch = self.rollout_buffer.sample(self.batch_size_stage2)
            self.optim_stage2.zero_grad()
            loss = 0.0
            for transition in batch:
                current_q = self.model.forward_critic_action(
                    self.device, transition.state, transition.action
                )["q_vm_critic"]
                with torch.no_grad():
                    target = torch.tensor(
                        transition.step_vm_cost, dtype=torch.float32, device=self.device
                    )
                    if not transition.done:
                        if transition.next_state is None:
                            raise ValueError(
                                "A non-terminal transition requires a next state."
                            )
                        outputs = self.model.forward_critic(
                            self.device, transition.next_state
                        )
                        action = self.policy.select_greedy_action(
                            outputs["subdeadline_risk_all"].cpu().numpy(),
                            outputs["q_vm_critic_all"].cpu().numpy(),
                            remove_vm=transition.next_state.remove_vm,
                        )
                        target = (
                            target
                            + self.gamma_vm_td
                            * self.model.forward_target_critic(outputs["z_all"])[action]
                        )
                loss = loss + F.huber_loss(current_q, target)
            loss = loss / len(batch)
            loss.backward()
            self._clip_if_needed(params)
            self.optim_stage2.step()
            soft_update(
                self.model.vm_critic_target, self.model.vm_critic, self.vm_target_tau
            )
            total_loss += float(loss.item())
        return {"stage2_vm_td_loss": total_loss / self.stage2_updates_per_iter}
