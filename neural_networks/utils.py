from typing import List, OrderedDict, Tuple
import torch
import torch.nn as nn


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
    elif "lettuce_model_multimodal_mobnetv3" == name:
        model = neural_networks.lettuce_model_multimodal_mobnetv3(**kwargs)
    elif "lettuce_model" == name:
        model = neural_networks.lettuce_model(**kwargs)
    elif "lettuce_model_mobnetv3" == name:
        model = neural_networks.lettuce_model_mobnetv3(**kwargs)
    elif "lettuce_model_no_height" == name:
        model = neural_networks.lettuce_model(with_height=False)
    elif "baseline_model_80" == name:
        model = neural_networks.baseline_model("80")
    elif "baseline_model_90" == name:
        model = neural_networks.baseline_model("90")
    elif "baseline_model_2"== name:
        model = neural_networks.baseline_model("2")
    else:
        raise ValueError(f"Unexpected model name, got: {name}")

    model.eval()

    return model


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
