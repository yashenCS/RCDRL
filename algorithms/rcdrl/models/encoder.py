import torch.nn as nn

from algorithms.rcdrl.models.gace_encoder import GACEEncoder


class ActionConditionedEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.backbone = GACEEncoder(config)
        self.embed_dim = self.backbone.output_dim

    def forward(self, device, state):
        return self.backbone(device, state)
