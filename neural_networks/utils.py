from typing import List, OrderedDict, Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.detection._utils import _topk_min
from torchvision.ops import boxes as box_ops

from neural_networks.custom_types import LettuceDetectionOutputs


def retrieve_out_channels(
        model: nn.Module,
        size: Tuple[int, int],
        dual_backbone=False
) -> List[int]:
    """
    This method retrieves the number of output channels of a specific model.

    Args:
        model (nn.Module): The model for which we estimate the out_channels.
            It should return a single Tensor or an OrderedDict[Tensor].
        size (Tuple[int, int]): The size (wxh) of the input.

    Returns:
        out_channels (List[int]): A list of the output channels of the model.
    """
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        if dual_backbone:
            features = model(tmp_img, tmp_img)
        else:
            features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels


def get_model(name: str, **kwargs) -> nn.Module:
    import neural_networks

    if "lettuce_model_multimodal" == name:
        model = neural_networks.lettuce_model_multimodal(**kwargs)
    elif "lettuce_model" == name:
        model = neural_networks.lettuce_model(**kwargs)
    elif "lettuce_model_no_height" == name:
        model = neural_networks.lettuce_model(with_height=False)
    elif "baseline_model_80" == name:
        model = neural_networks.baseline_model("80")
    elif "baseline_model_90" == name:
        model = neural_networks.baseline_model("90")
    elif "baseline_model_2" == name:
        model = neural_networks.baseline_model("2")
    else:
        raise ValueError(f"Unexpected model name, got: {name}")

    model.eval()

    return model


def postprocess_lettuce_detections(
        head_outputs: LettuceDetectionOutputs,
        image_shapes: List[Tuple[int, int]],
        box_coder,
        score_thresh,
        phenotype_stds,
        phenotype_means,
        topk_candidates=400,
        detections_per_img: int = 200,
        nms_thresh: float = 0.5,
) -> List[Dict[str, Tensor]]:
    bbox_regression = head_outputs.bbox_regression
    pred_scores = F.softmax(head_outputs.cls_logits, dim=-1)
    phenotypes_pred = head_outputs.phenotypes_pred
    image_anchors = head_outputs.anchors

    num_classes = pred_scores.size(-1)
    device = pred_scores.device

    detections: List[Dict[str, Tensor]] = []

    for boxes, scores, phenotypes, anchors, image_shape in zip(bbox_regression, pred_scores, phenotypes_pred,
                                                               image_anchors, image_shapes):
        boxes = box_coder.decode_single(boxes, anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        image_boxes = []
        image_scores = []
        image_labels = []
        image_phenotypes = []
        for label in range(1, num_classes):
            score = scores[:, label]

            keep_idxs = score > score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]
            phenotype = phenotypes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = _topk_min(score, topk_candidates, 0)
            score, idxs = score.topk(num_topk)
            box = box[idxs]
            phenotype = (phenotype[idxs] * phenotype_stds) + phenotype_means  # denormalize

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
            image_phenotypes.append(phenotype)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_phenotypes = torch.cat(image_phenotypes, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        keep = keep[: detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
                "phenotypes": image_phenotypes[keep]
            }
        )
    return detections


def list_to_tensor_stack(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Converts a list of tensors into a single tensor by stacking them along a new
    first dimension. All tensors in the list must have the same shape.

    Args:
        tensor_list (List[torch.Tensor]): A list of PyTorch tensors.
                                          All tensors must have the same shape.

    Returns:
        torch.Tensor: A single tensor where the first dimension is N (the
                      number of tensors in the input list), and the subsequent
                      dimensions match the shape of the individual input tensors.
                      Shape: [N, *tensor_list[0].shape].

    Raises:
        ValueError: If the input list is empty or if tensors in the list
                    do not all have the same shape.
    """
    if not tensor_list:
        raise ValueError("Input tensor_list cannot be empty for stacking.")

    # Check if all tensors have the same shape
    # (torch.stack will also raise an error, but this gives a clearer message)
    first_tensor_shape = tensor_list[0].shape
    for i, tensor in enumerate(tensor_list[1:], start=1): # Start enumeration from 1 for message
        if tensor.shape != first_tensor_shape:
            raise ValueError(
                f"All tensors in the list must have the same shape. "
                f"Shape of tensor at index 0: {first_tensor_shape}, "
                f"but shape of tensor at index {i}: {tensor.shape}"
            )

    # Stack the tensors along a new dimension (dim=0 makes N the first dimension)
    # The resulting tensor will have shape [N, original_dim1, original_dim2, ...]
    stacked_tensor = torch.stack(tensor_list, dim=0)
    return stacked_tensor
