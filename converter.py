import torch
import torch._dynamo as dynamo
from torchvision import models

from neural_networks import lettuce_model

def convert():
    model = lettuce_model()
    chkpt = torch.load("models/instances_subset_50epc_patched.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(chkpt["model"])
    model.eval()

    trace_inputs = (torch.randn(1, 3, 320, 320),)
    traced_model = torch.jit.trace(model, trace_inputs)

    output = traced_model(trace_inputs)
    print(output)

    # exported = torch.export.export(model, sample_inputs)
    # print(exported)


if __name__ == "__main__":
    convert()