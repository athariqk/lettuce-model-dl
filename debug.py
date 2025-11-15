from torchinfo import summary
from neural_networks import lettuce_model_multimodal_mobnetv3
model = lettuce_model_multimodal_mobnetv3()
input_shape = (1, 3, 320, 320)
print(summary(model, input_size=input_shape, depth=6))