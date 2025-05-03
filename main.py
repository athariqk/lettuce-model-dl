from torchinfo import summary
from models import ssdlite_mobilevit

model = ssdlite_mobilevit()
model.eval()

print(summary(model, input_size=(32, 3, 320, 320)))
