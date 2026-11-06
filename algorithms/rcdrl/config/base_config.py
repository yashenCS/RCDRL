from abc import ABCMeta
import argparse
import os
import yaml


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_scheduling_rcdrl.yaml")


class BaseConfigRCDRL(metaclass=ABCMeta):


    def __init__(self):
        parser = argparse.ArgumentParser(description="Arguments for the standalone RCDRL training pipeline")
        parser.add_argument(
            "--config",
            type=str,
            default=DEFAULT_CONFIG_PATH,
            help="Path to the RCDRL YAML config file.",
        )
        parser.add_argument("--seed", type=int, default=None, help="Override the YAML random seed for this training run.")
        parser.add_argument("--device", type=str, default=None, help="Override the execution device.")
        args = parser.parse_args()


        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)


        if args.seed is not None:
            config.setdefault("env", {})["seed"] = int(args.seed)
        if args.device is not None:
            config.setdefault("runtime", {})["device"] = args.device

        self.config = {
            "runtime-config": vars(args),
            "yaml-config": config,
        }
