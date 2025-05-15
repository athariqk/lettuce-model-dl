from typing import List, OrderedDict, Tuple
import torch
import torch.nn as nn

def retrieve_out_channels(
    model: nn.Module,
    size: Tuple[int, int],
    dual_backbone = False
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

    if "lettuce_model_multimodal" in name:
        model = neural_networks.lettuce_model_multimodal(**kwargs)
    elif "lettuce_model_unimodal" in name:
        model = neural_networks.lettuce_model_unimodal(**kwargs)
    elif "lettuce_model" in name:
        model = neural_networks.lettuce_model(**kwargs)
    elif "baseline_model" in name:
        model = neural_networks.baseline_model()
    else:
        raise ValueError("Unexpected model name: {name}")
    
    model.eval()

    return model
