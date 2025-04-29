import torch
from torchinfo import summary
from coco_utils import get_coco
from engine import evaluate
from models import ssdlite_mobilevit
from train import get_transform

model = ssdlite_mobilevit()
model.eval()

weights = torch.load("output/model_0.pth")
model.load_state_dict(weights)
