from env.workflow_scheduling_v3.simulator_wf import WFEnv
from algorithms.rcdrl.models.model import RCDRLNet
from algorithms.rcdrl.core.rollout_buffer import RolloutBuffer
from algorithms.rcdrl.core.trainer import RCDRLTrainer
from algorithms.rcdrl.config.validation import validate_rcdrl_config


class BuilderRCDRL:


    def __init__(self, device, baseconfig, train_Set_setting, test_Set_setting):
        self.device = device
        self.config = baseconfig
        self.train_Set_setting = train_Set_setting
        self.test_Set_setting = test_Set_setting

    def build(self):

        yaml_cfg = self.config.config["yaml-config"]

        train_env_cfg = dict(yaml_cfg["env"])
        train_env_cfg["gamma"] = train_env_cfg["gamma_train"]
        env = WFEnv(self.device, train_env_cfg["name"], train_env_cfg, self.train_Set_setting, self.test_Set_setting)

        test_env_cfg = dict(yaml_cfg["env"])
        test_env_cfg["gamma"] = test_env_cfg["gamma_test"]
        test_env_cfg["wf_num"] = test_env_cfg["wf_num_test"]
        env_test = WFEnv(self.device, test_env_cfg["name"], test_env_cfg, self.train_Set_setting, self.test_Set_setting)

        validate_rcdrl_config(yaml_cfg, self.train_Set_setting, self.test_Set_setting)

        model = RCDRLNet(yaml_cfg)
        rollout_buffer = RolloutBuffer()

        return RCDRLTrainer(
            device=self.device,
            config=self.config,
            env=env,
            env_test=env_test,
            model=model,
            rollout_buffer=rollout_buffer,
            train_Set_setting=self.train_Set_setting,
            test_Set_setting=self.test_Set_setting,
        )
