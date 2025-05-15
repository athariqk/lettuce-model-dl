import torch
from torchinfo import summary
from neural_networks import lettuce_model, lettuce_model_multimodal, lettuce_model_unimodal
from cvnets.models.detection.ssd import SingleShotMaskDetector
from torchvision.models.detection.ssd import SSD

# model = ssdlite_mobilevit_multimodal()
# model.eval()

# batch_size = 4
# modality1_channels = 3  # e.g., RGB image
# modality2_channels = 3  # e.g., Depth image
# height = 320
# width = 320

# input_shape_modality1 = (batch_size, modality1_channels, height, width)
# input_shape_modality2 = (batch_size, modality2_channels, height, width)

# print(summary(model, input_size=[input_shape_modality1, input_shape_modality2]))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = estimator_model()
# model.to(device)
# model.eval()

model: SingleShotMaskDetector = torch.load(
            "models/coco-ssd-mobilevitv2-0.75_structure.pt", weights_only=False)
checkpoint = torch.load("models/coco-ssd-mobilevitv2-0.75_weight.pt")

# # # Step 3.
# # for k in model.state_dict().keys():
# #     if model.state_dict()[k].shape != checkpoint[k].shape:
# #         print('key {} will be removed, orishape: {}, training shape: {}'.format(k, checkpoint[k].shape, model.state_dict()[k].shape))
# #         checkpoint.pop(k)

model.load_state_dict(checkpoint, strict=False)
model.eval()

torch.save(model, "models/coco-ssd-mobilevitv2-0.75_pretrained.pt")

# batch_size = 1
# channels = 3
# height = 320
# width = 320

# dummy_input_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.float32).to(device)

# print(model(dummy_input_tensor))

# input_shape = (batch_size, channels, height, width)

# print(summary(model, input_size=input_shape, depth=6))
