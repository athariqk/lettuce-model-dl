from typing import NamedTuple

import torch


class LettuceDetectionOutputs(NamedTuple):
    cls_logits: torch.Tensor      # Expected shape: [B, N, NumClasses]
    bbox_regression: torch.Tensor # Expected shape: [B, N, 4]
    phenotypes_pred: torch.Tensor # Expected shape: [B, N, NumPhenotypes]
    anchors: torch.Tensor         # Expected shape: [N, A]
