import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.rcdrl.models.encoder import ActionConditionedEncoder


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RCDRLNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = ActionConditionedEncoder(config["encoder"])
        embed_dim = self.encoder.embed_dim
        hidden_dim = int(config["model"]["hidden_dim"])
        self.deadline_head = MLPHead(embed_dim, hidden_dim)
        self.vm_critic = MLPHead(embed_dim, hidden_dim)
        self.vm_critic_target = copy.deepcopy(self.vm_critic)

    def encode_all(self, device, state):
        return self.encoder(device, state)

    def _deadline_outputs(self, z_all: torch.Tensor) -> dict:
        logits = self.deadline_head(z_all).squeeze(-1)
        return {
            "subdeadline_logit_all": logits,
            "subdeadline_risk_all": torch.sigmoid(logits),
        }

    def forward_stage1(self, device, state):
        z_all = self.encode_all(device, state)["z_all"]
        return {"z_all": z_all, **self._deadline_outputs(z_all)}

    def forward_critic(self, device, state):
        z_all = self.encode_all(device, state)["z_all"]
        return {
            "z_all": z_all,
            **self._deadline_outputs(z_all),
            "q_vm_critic_all": F.softplus(self.vm_critic(z_all)).squeeze(-1),
        }

    def forward_target_critic(self, z_all: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.vm_critic_target(z_all)).squeeze(-1)

    def forward_critic_action(self, device, state, action: int):
        outputs = self.forward_critic(device, state)
        return {"q_vm_critic": outputs["q_vm_critic_all"][int(action)]}
