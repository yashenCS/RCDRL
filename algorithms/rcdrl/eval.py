if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os

import torch
import yaml

from config.test_set_config import testSet_Generate
from config.train_set_config import trainSet_Generate
from algorithms.rcdrl.core.reference import FastestNewVMEvaluator
from algorithms.rcdrl.core.reference import FastestNewVMPolicy
from algorithms.rcdrl.builder import BuilderRCDRL
from algorithms.rcdrl.config.eval_config import EvalConfigRCDRL
from algorithms.rcdrl.core.selection import constraint_metrics
from algorithms.rcdrl.core.utils import ensure_dir, set_global_seed, to_serializable


def _reference_result(stats, test_seed: int) -> dict:
    return {
        "split": "test_reference",
        "fixed_seed": int(test_seed),
        "episodes": len(stats.gamma_per_instance),
        "workflow_count": stats.wf_count,
        "deadline_met_count": stats.deadline_met_wf_count,
        "deadline_missed_count": stats.deadline_missed_wf_count,
        "success_rate": stats.deadline_met_wf_rate,
        "miss_rate": stats.deadline_missed_wf_rate,
        "mean_vm_cost_per_episode": stats.avg_vm_cost,
        "total_vm_cost": stats.vm_cost_total,
        "mean_environment_reward": stats.avg_return,
    }


def _policy_result(row: dict, reference_success: float, relative_deadline_gap: float) -> dict:
    protocol = constraint_metrics(
        success_rate=row["deadline_met_wf_rate"],
        reference_success_rate=reference_success,
        relative_deadline_gap=relative_deadline_gap,
    )
    return {
        "split": "test",
        "episodes": len(json.loads(row["gamma_per_instance"])),
        "workflow_count": row["wf_count"],
        "deadline_met_count": row["deadline_met_wf_count"],
        "deadline_missed_count": row["deadline_missed_wf_count"],
        "success_rate": row["deadline_met_wf_rate"],
        "miss_rate": row["deadline_missed_wf_rate"],
        "mean_vm_cost_per_episode": row["VM_cost"],
        "total_vm_cost": row["vm_cost_total"],
        "mean_environment_reward": row["rewards"],
        "mean_vm_exec_hour": row["VM_execHour"],
        "mean_vm_total_hour": row["VM_totHour"],
        "mean_deadline_penalty": row["deadline_penalty"],
        "mean_missed_workflows_per_episode": row["missDeadlineNum"],
        "normalized_lateness_mean_all_workflows": row["norm_lateness_mean_all_wf"],
        "normalized_lateness_mean_missed_workflows": row["norm_lateness_mean_missed_wf"],
        **protocol,
    }


def _write_json_atomic(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def main() -> None:

    eval_config = EvalConfigRCDRL()
    runtime_cfg = eval_config.config["runtime-config"]
    yaml_config = eval_config.config["yaml-config"]
    yaml_path = runtime_cfg["config"]
    policy_path = runtime_cfg["policy_path"]

    if runtime_cfg["trained_workflow_size"] != "S":
        raise ValueError(
            "Protocol violation: final S/M/L testing requires a checkpoint trained on S, "
            f"but profile.yaml reports {runtime_cfg['trained_workflow_size']}."
        )
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Selected checkpoint not found: {policy_path}")

    training_seed = int(runtime_cfg["training_seed"])
    test_seed = int(yaml_config.get("evaluation", {}).get("test_seed", 42))
    relative_deadline_gap = float(yaml_config.get("constraint", {}).get("relative_deadline_gap", 0.10))
    set_global_seed(test_seed)
    train_set_setting = trainSet_Generate(yaml_path)
    test_set_setting = testSet_Generate(yaml_path, seed=test_seed)
    set_global_seed(test_seed)

    requested_device = runtime_cfg.get("device") or yaml_config.get("runtime", {}).get("device")
    device = requested_device if requested_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}", flush=True)
    print(f"Training seed: {training_seed}", flush=True)
    print(f"Fixed test seed: {test_seed}", flush=True)
    print(f"Trained on S; testing on: {runtime_cfg['wf_size']}", flush=True)
    print(f"Selected checkpoint: {policy_path}", flush=True)
    print("\n================ EFFECTIVE RCDRL TEST CONFIG ================", flush=True)
    print(yaml.dump(yaml_config, sort_keys=False, allow_unicode=True), flush=True)
    print("==============================================================\n", flush=True)

    trainer = BuilderRCDRL(device, eval_config, train_set_setting, test_set_setting).build()
    reference_stats = FastestNewVMEvaluator(
        env_test=trainer.env_test,
        policy=FastestNewVMPolicy(),
        config=yaml_config,
    ).evaluate(n_episodes=int(yaml_config["env"]["validNum"]), outer_iter=0)
    reference = _reference_result(reference_stats, test_seed)
    row = trainer.eval(
        policy_path=policy_path,
        outer_iter=int(runtime_cfg["model_num"]) - 1,
        n_episodes=int(yaml_config["env"]["validNum"]),
    )
    result = _policy_result(row, reference["success_rate"], relative_deadline_gap)

    with open(runtime_cfg["selection_path"], "r", encoding="utf-8") as f:
        selection = json.load(f)
    output = {
        "algorithm": "RCDRL",
        "trained_workflow_size": "S",
        "evaluated_workflow_size": runtime_cfg["wf_size"],
        "training_seed": training_seed,
        "fixed_test_seed": test_seed,
        "dynamic_gamma": bool(yaml_config["env"].get("dynamic_gamma", False)),
        "dynamic_gamma_wf": bool(yaml_config["env"].get("dynamic_gamma_wf", False)),
        "checkpoint_model_num": int(runtime_cfg["model_num"]),
        "checkpoint": policy_path,
        "validation_selection": selection["validation"],
        "test_reference": reference,
        "test_result": result,
    }
    _write_json_atomic(runtime_cfg["output"], output)

    print(
        f"EVALUATED RCDRL trained_on=S tested_on={runtime_cfg['wf_size']} "
        f"seed={training_seed} success={result['success_rate']:.4f} "
        f"cost={result['mean_vm_cost_per_episode']:.3f} "
        f"D-Gap={result['deadline_gap']:.4f} feasible={result['feasible']}",
        flush=True,
    )
    print(f"Saved final test result to: {runtime_cfg['output']}", flush=True)


if __name__ == "__main__":
    main()
