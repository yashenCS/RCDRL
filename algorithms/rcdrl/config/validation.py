from typing import Any
import numpy as np


def _as_int(section: dict, key: str) -> int:
    return int(section[key])


def _require_positive_int(section_name: str, section: dict, key: str) -> None:
    value = _as_int(section, key)
    if value <= 0:
        raise ValueError(
            f"{section_name}.{key} must be a positive integer, but got {value}."
        )


def _require_non_negative_int(section_name: str, section: dict, key: str) -> None:
    value = _as_int(section, key)
    if value < 0:
        raise ValueError(
            f"{section_name}.{key} must be a non-negative integer, but got {value}."
        )


def _require_unit_interval(section_name: str, section: dict, key: str) -> None:
    value = float(section[key])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{section_name}.{key} must lie in [0, 1], but got {value}.")


def _validate_dataset_shape(
    name: str, data: Any, expected_instances: int = None
) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape [group, instance, workflow], but got {array.shape}."
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(
            f"{name} must be non-empty on every axis, but got {array.shape}."
        )
    if expected_instances is not None and int(array.shape[1]) != int(
        expected_instances
    ):
        raise ValueError(
            f"{name} instance axis mismatch: expected {expected_instances}, but got shape {array.shape}."
        )
    return array


def validate_rcdrl_config(yaml_cfg: dict, train_set_setting, test_set_setting) -> None:
    env = yaml_cfg["env"]
    training = yaml_cfg["training"]
    optim = yaml_cfg["optim"]
    gap = float(yaml_cfg.get("constraint", {}).get("relative_deadline_gap", 0.1))
    if not 0.0 <= gap < 1.0:
        raise ValueError("constraint.relative_deadline_gap must lie in [0, 1).")
    evaluation = yaml_cfg.get("evaluation", {})
    if int(evaluation.get("validation_seed", 41)) == int(
        evaluation.get("test_seed", 42)
    ):
        raise ValueError("Validation and test seeds must differ.")
    _require_positive_int("model", yaml_cfg["model"], "hidden_dim")
    for key in ["wf_num", "wf_num_test", "evalNum", "validNum"]:
        _require_positive_int("env", env, key)
    _require_non_negative_int("env", env, "dataGen")
    for key in [
        "outer_iters",
        "rollout_episodes_per_iter",
        "stage1_updates_per_iter",
        "stage2_updates_per_iter",
        "batch_size_stage1",
        "batch_size_stage2",
        "eval_freq",
        "save_freq",
    ]:
        _require_positive_int("training", training, key)
    for key in ["delta", "epsilon"]:
        _require_unit_interval("training", training, key)
    for key in ["gamma_vm_td", "vm_target_tau", "fallback_top_ratio"]:
        value = float(training[key])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"training.{key} must lie in (0, 1].")
    if training.get("grad_clip") is not None and float(training["grad_clip"]) <= 0.0:
        raise ValueError("training.grad_clip must be positive when specified.")
    for key in ["lr_stage1", "lr_stage2"]:
        if float(optim[key]) <= 0.0:
            raise ValueError(f"optim.{key} must be positive.")
    if float(optim.get("weight_decay", 0.0)) < 0.0:
        raise ValueError("optim.weight_decay must be non-negative.")
    for prefix, settings, instances in [
        ("train", train_set_setting, env["evalNum"]),
        ("test", test_set_setting, env["validNum"]),
    ]:
        matrix_name = prefix + "Matrix"
        matrix = _validate_dataset_shape(
            matrix_name, getattr(settings, matrix_name), expected_instances=instances
        )
        arrivals = np.asarray(getattr(settings, prefix + "ArrivalTime"))
        if arrivals.shape != matrix.shape:
            raise ValueError(
                f"{prefix}ArrivalTime must have shape {matrix.shape}, got {arrivals.shape}."
            )
        if bool(env.get("dynamic_gamma", False)):
            per_workflow = bool(env.get("dynamic_gamma_wf", False))
            gamma_name = prefix + (
                "_dynamic_gamma_wf" if per_workflow else "_dynamic_gamma"
            )
            gammas = np.asarray(getattr(settings, gamma_name))
            expected = matrix.shape if per_workflow else matrix.shape[:2]
            if gammas.shape != expected:
                raise ValueError(
                    f"{gamma_name} must have shape {expected}, got {gammas.shape}."
                )
