from torchinfo import summary
from neural_networks import ssdlite_mobilevit_multimodal, ssdlite_mobilevit_unimodal

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

model = ssdlite_mobilevit_unimodal()
model.eval()

batch_size = 4
channels = 3
height = 320
width = 320

input_shape = (batch_size, channels, height, width)

print(summary(model, input_size=input_shape))
