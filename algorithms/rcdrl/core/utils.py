import copy
import os
import random
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
from algorithms.rcdrl.core.types import RCDRLState


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def parse_env_state(state_dict_list: dict) -> RCDRLState:
    state = state_dict_list["0"]
    remove_vm = state_dict_list.get("removeVM", None)
    return RCDRLState(
        ob=np.array(state["state"], copy=True),
        dag=copy.deepcopy(state["DAG"]),
        node_id=int(state["Node_id"]),
        vm_features=np.array(state["VM_configuration"], copy=True),
        gace_dag_node_features=None
        if "gace_dag_node_features" not in state
        else np.array(state["gace_dag_node_features"], dtype=np.float32, copy=True),
        active_wf_features=None
        if "active_wf_features" not in state
        else np.array(state["active_wf_features"], dtype=np.float32, copy=True),
        candidate_action_features=None
        if "candidate_action_features" not in state
        else np.array(state["candidate_action_features"], dtype=np.float32, copy=True),
        global_features=None
        if "global_features" not in state
        else np.array(state["global_features"], dtype=np.float32, copy=True),
        remove_vm=None if remove_vm is None else int(remove_vm),
    )


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    module.train()
    for param in module.parameters():
        param.requires_grad = True


@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


@torch.no_grad()
def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def _require_instance_axis(name: str, data: np.ndarray) -> Tuple[int, int]:
    if data.ndim < 2:
        raise ValueError(
            f"{name} must have at least two axes: [group, instance, ...], but got shape {data.shape}."
        )
    return (int(data.shape[0]), int(data.shape[1]))


def resolve_env_episode_indices(
    env, config: dict, mode: str, outer_iter: int, ep_num: int
) -> Tuple[int, int]:
    if mode not in {"train", "test"}:
        raise ValueError(f"Unsupported episode collection mode: {mode}")
    if mode == "train":
        data = np.asarray(env.train_Set_setting.trainMatrix)
        num_groups, num_instances = _require_instance_axis("trainMatrix", data)
        episodes_per_iter = int(config["training"]["rollout_episodes_per_iter"])
        flat_episode_id = outer_iter * episodes_per_iter + ep_num
        group_idx = flat_episode_id // num_instances % num_groups
        instance_idx = flat_episode_id % num_instances
        return (int(group_idx), int(instance_idx))
    data = np.asarray(env.test_Set_setting.testMatrix)
    _, num_instances = _require_instance_axis("testMatrix", data)
    return (0, int(ep_num % num_instances))


def get_gamma_for_episode(env, config: dict, mode: str, outer_iter: int, ep_num: int):
    env_cfg = config["env"]
    if not env_cfg.get("dynamic_gamma", False):
        return env_cfg["gamma_train"] if mode == "train" else env_cfg["gamma_test"]
    group_idx, instance_idx = resolve_env_episode_indices(
        env, config, mode, outer_iter, ep_num
    )
    if mode == "train":
        if env_cfg.get("dynamic_gamma_wf", False):
            return env.train_Set_setting.train_dynamic_gamma_wf[group_idx][instance_idx]
        return env.train_Set_setting.train_dynamic_gamma[group_idx][instance_idx]
    if env_cfg.get("dynamic_gamma_wf", False):
        return env.test_Set_setting.test_dynamic_gamma_wf[group_idx][instance_idx]
    return env.test_Set_setting.test_dynamic_gamma[group_idx][instance_idx]


def filtered_feasible_actions(action_num: int, remove_vm: Optional[int]) -> List[int]:
    feasible = list(range(action_num))
    if remove_vm is not None and remove_vm in feasible:
        feasible.remove(int(remove_vm))
    if not feasible:
        feasible = list(range(action_num))
    return feasible
