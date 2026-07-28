"""LitePT backbone plus binary segmentation head for native 6D inputs."""

import torch.nn as nn

import config as C
from litept6d.model import LitePT


class LitePTBinarySeg(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        cfg = C.get_model_config()
        final_dim = cfg.pop("FINAL_DIM")
        cfg["coord_dim"] = int(getattr(C, "NATIVE_COORD_DIM", 6))
        cfg["activation_checkpointing"] = bool(getattr(C, "ACTIVATION_CHECKPOINTING", False))
        self.backbone = LitePT(in_channels=in_channels, enc_mode=False, **cfg)
        self.head = nn.Linear(final_dim, 2)

    def forward(self, data_dict, return_dense: bool = False):
        point = self.backbone(data_dict)
        logits_sampled = self.head(point.feat)
        if return_dense:
            logits_dense = logits_sampled[data_dict["inverse"]]
            return logits_sampled, logits_dense, point
        return logits_sampled, point
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
