import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from neural_networks.utils import list_to_tensor_stack


class DefaultBoxGenerator(nn.Module):
    """
    This module generates the default boxes of SSD for a set of feature maps and image sizes.

    Args:
        aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
        min_ratio (float): The minimum scale :math:`\text{s}_{\text{min}}` of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        max_ratio (float): The maximum scale :math:`\text{s}_{\text{max}}`  of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
            the ``min_ratio`` and ``max_ratio`` parameters.
        steps (List[int]], optional): It's a hyper-parameter that affects the tiling of default boxes. If not provided
            it will be estimated from the data.
        clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
            is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
    """

    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ):
        super().__init__()
        if steps is not None and len(aspect_ratios) != len(steps):
            raise ValueError("aspect_ratios and steps should have the same length")
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        num_outputs = len(aspect_ratios)

        # Estimation of default boxes scales
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)]
                self.scales.append(1.0)
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            self.scales = scales

        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(
        self, num_outputs: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
    ) -> List[Tensor]:
        _wh_pairs: List[Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
        return _wh_pairs

    def num_anchors_per_location(self) -> List[int]:
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(
        self, grid_sizes: List[List[int]], image_size: List[int], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k = image_size[1] / self.steps[k]
                y_f_k = image_size[0] / self.steps[k]
            else:
                y_f_k, x_f_k = f_k

            shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
            shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y) * self._wh_pairs[k].shape[0], dim=-1).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

            default_box = torch.cat((shifts, wh_pairs), dim=1)

            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"aspect_ratios={self.aspect_ratios}"
            f", clip={self.clip}"
            f", scales={self.scales}"
            f", steps={self.steps}"
            ")"
        )
        return s

    def forward(self, images: Tensor, feature_maps: List[Tensor]) -> Tensor:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = images.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        x_y_size = torch.tensor([image_size[1], image_size[0]], device=default_boxes.device)
        for _ in range(images.shape[0]):
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat(
                [
                    (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                    (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                ],
                -1,
            )
            dboxes.append(dboxes_in_image)
        return list_to_tensor_stack(dboxes)
