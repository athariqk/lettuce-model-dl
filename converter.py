from typing import Dict, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile

# import ai_edge_torch

from neural_networks import lettuce_model, lettuce_regressor_model

def assertion_check(
    py_out: Union[Tensor, Dict, Tuple], jit_out: Union[Tensor, Dict, Tuple]
) -> None:
    if isinstance(py_out, Dict):
        assert isinstance(jit_out, Dict)
        keys = py_out.keys()
        for k in keys:
            np.testing.assert_almost_equal(
                py_out[k].cpu().numpy(),
                jit_out[k].cpu().numpy(),
                decimal=3,
                verbose=True,
            )
    elif isinstance(py_out, Tensor):
        assert isinstance(jit_out, Tensor)
        np.testing.assert_almost_equal(
            py_out.cpu().numpy(), jit_out.cpu().numpy(), decimal=3, verbose=True
        )
    elif isinstance(py_out, Tuple):
        assert isinstance(jit_out, Tuple)
        for x, y in zip(py_out, jit_out):
            np.testing.assert_almost_equal(
                x.cpu().numpy(), y.cpu().numpy(), decimal=3, verbose=True
            )

    else:
        raise NotImplementedError(
            "Only Dictionary[Tensors] or Tuple[Tensors] or Tensors are supported as outputs"
        )


def convert():
    pytorch_model = lettuce_model()
    chkpt = torch.load(
        "models/singlebb_baseline_detection_30_epochs_patched.pth", map_location="cpu", weights_only=False
    )
    pytorch_model.load_state_dict(chkpt["model"])

    input_tensor = torch.randint(
        low=0,
        high=255,
        size=(1, 3, 320, 320),
        device="cpu",
    )
    input_tensor = input_tensor.float().div(255.0)

    if pytorch_model.training:
        pytorch_model.eval()

    with torch.no_grad():
        pytorch_out = pytorch_model(input_tensor)

        jit_model = torch.jit.trace(pytorch_model, input_tensor)
        jit_out = jit_model(input_tensor)
        assertion_check(py_out=pytorch_out, jit_out=jit_out)

        jit_model_optimized = optimize_for_mobile(jit_model)
        jit_optimzied_out = jit_model_optimized(input_tensor)
        assertion_check(py_out=pytorch_out, jit_out=jit_optimzied_out)

        jit_model_optimized._save_for_lite_interpreter("models/singlebb_baseline_detection_30_epochs_patched_jit.pt")

def dummy():
    model = lettuce_regressor_model()
    sample_inputs = (torch.randn(1, 3, 320, 320),)
    exported = torch.jit.trace(model, sample_inputs)
    print(exported)


if __name__ == "__main__":
    convert()
    # dummy()