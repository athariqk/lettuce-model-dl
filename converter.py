import torch
import ai_edge_torch

from neural_networks import lettuce_model

def convert():
    model = lettuce_model()
    chkpt = torch.load(
        "models/singlebb_baseline_detection_30_epochs_patched.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(chkpt["model"])
    model.eval()

    sample_inputs = (torch.randn(1, 3, 320, 320),)
    edge_model = ai_edge_torch.convert(model, sample_inputs)

    edge_output = edge_model(*sample_inputs)
    print(edge_output)

    # exported = torch.export.export(model, sample_inputs)
    # print(exported)


if __name__ == "__main__":
    convert()