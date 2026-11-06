if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from algorithms.rcdrl.builder import BuilderRCDRL
from algorithms.rcdrl.config.base_config import BaseConfigRCDRL
from config.test_set_config import testSet_Generate
from config.train_set_config import trainSet_Generate
from algorithms.rcdrl.core.utils import set_global_seed


def main() -> None:


    baseconfig = BaseConfigRCDRL()
    yaml_config = baseconfig.config["yaml-config"]
    yaml_path = baseconfig.config["runtime-config"]["config"]


    run_seed = int(yaml_config["env"]["seed"])
    set_global_seed(run_seed)


    train_set_setting = trainSet_Generate(yaml_path)
    validation_seed = int(yaml_config.get("evaluation", {}).get("validation_seed", 41))
    test_set_setting = testSet_Generate(yaml_path, seed=validation_seed)


    set_global_seed(run_seed)


    requested_device = baseconfig.config["runtime-config"].get("device") or yaml_config.get("runtime", {}).get("device")
    device = requested_device if requested_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}", flush=True)
    print(f"Using seed: {run_seed}", flush=True)
    print(f"Using validation seed: {validation_seed}", flush=True)

    print(f"Config path: {yaml_path}", flush=True)
    print(f"Runtime args: {baseconfig.config['runtime-config']}", flush=True)
    print("\n================ EFFECTIVE RCDRL CONFIG ================", flush=True)
    print(yaml.dump(yaml_config, sort_keys=False, allow_unicode=True, default_flow_style=False), flush=True)
    print("========================================================\n", flush=True)


    trainer = BuilderRCDRL(device, baseconfig, train_set_setting, test_set_setting).build()
    trainer.train()


if __name__ == "__main__":
    main()
