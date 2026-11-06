from abc import ABCMeta
import argparse
import json
import os

import yaml


class EvalConfigRCDRL(metaclass=ABCMeta):


    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Evaluate one selected S-trained RCDRL checkpoint on S, M, or L."
        )
        parser.add_argument(
            "--log_path",
            type=str,
            required=True,
            help="Completed run directory containing profile.yaml and selection.json.",
        )
        parser.add_argument(
            "--model_num",
            "-m",
            type=int,
            default=None,
            help="Checkpoint number. Omit to use selection.json automatically.",
        )
        parser.add_argument("--wf_size", "-w", choices=["S", "M", "L"], required=True)
        parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
        parser.add_argument("--device", type=str, default=None, help="Override the execution device.")
        args = parser.parse_args()

        log_path = os.path.abspath(args.log_path)
        config_path = os.path.join(log_path, "profile.yaml")
        selection_path = os.path.join(log_path, "selection.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        trained_workflow_size = str(config["env"]["wf_size"])
        training_seed = int(config["env"]["seed"])
        evaluation_cfg = config.get("evaluation", {})
        if int(evaluation_cfg.get("validation_seed", -1)) != 41 or int(evaluation_cfg.get("test_seed", -1)) != 42:
            raise ValueError("Final testing requires validation seed 41 and test seed 42.")
        if not bool(config["env"].get("dynamic_gamma", False)) or not bool(config["env"].get("dynamic_gamma_wf", False)):
            raise ValueError("Final testing requires workflow-level dynamic gamma.")
        if float(config.get("constraint", {}).get("relative_deadline_gap", -1.0)) != 0.10:
            raise ValueError("Final testing requires constraint.relative_deadline_gap=0.10.")
        if not os.path.isfile(selection_path):
            raise FileNotFoundError(
                f"Completed selection not found: {selection_path}. Finish training before final testing."
            )
        with open(selection_path, "r", encoding="utf-8") as f:
            selection = json.load(f)
        if selection.get("status") != "complete":
            raise ValueError(f"RCDRL selection is not complete: {selection_path}")
        if int(selection.get("training_seed", -1)) != training_seed:
            raise ValueError(f"Training seed mismatch between profile.yaml and {selection_path}")
        if str(selection.get("trained_workflow_size")) != trained_workflow_size:
            raise ValueError(f"Training workflow-size mismatch in {selection_path}")
        if int(selection.get("completed_outer_iters", -1)) != int(config["training"]["outer_iters"]):
            raise ValueError(f"Incomplete training iteration count in {selection_path}")
        if int(selection.get("validation_seed", -1)) != int(evaluation_cfg["validation_seed"]):
            raise ValueError(f"Validation seed mismatch in {selection_path}")

        model_num = args.model_num
        if model_num is None:
            model_num = int(selection["selected_model_num"])
        elif int(model_num) != int(selection["selected_model_num"]):
            raise ValueError(
                f"Final testing only accepts selected model {selection['selected_model_num']}, "
                f"but got {model_num}."
            )

        policy_path = os.path.join(log_path, "saved_models", f"ep_{int(model_num)}.pt")
        config.setdefault("env", {})["wf_size"] = args.wf_size
        config["env"]["model_num"] = int(model_num)

        if args.device is not None:
            config.setdefault("runtime", {})["device"] = args.device
        config.setdefault("runtime", {})["log"] = True

        output_path = args.output or os.path.join(log_path, "cross_scale_test", f"evaluation_{args.wf_size}.json")
        self.config = {
            "runtime-config": {
                "config": config_path,
                "policy_path": policy_path,
                "selection_path": selection_path,
                "log_path": log_path,
                "model_num": int(model_num),
                "wf_size": args.wf_size,
                "device": args.device,
                "output": os.path.abspath(output_path),
                "training_seed": training_seed,
                "trained_workflow_size": trained_workflow_size,
            },
            "yaml-config": config,
        }
