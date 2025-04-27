import torch
from torchinfo import summary
from models import ssdlite_mobilevit

model = ssdlite_mobilevit()
model.eval()
summary(model, input_size=(34, 3, 320, 320))
