import argparse
import glob
import os
from typing import List, Optional
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from cvnets.models.detection.ssd import SingleShotMaskDetector, DetectionPredTuple
from matplotlib import patches
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import functional as F_vision
from torch.nn import functional as F
from torch import nn

from neural_networks import lettuce_model, LettuceModelEval


def get_arguments():
    parser = argparse.ArgumentParser(description="SSDLite-MobileViT Inference")
    parser.add_argument("--model-path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--weights", type=str, help="Path to the weights to load")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--lettuce-model", action="store_true")
    return parser.parse_args()


coco_names = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

lettuce_names = [
    "background",
    "lettuce"
]


class Colormap(object):
    """
    Generate colormap for visualizing segmentation masks or bounding boxes.

    This is based on the MATLab code in the PASCAL VOC repository:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def __init__(self, n: Optional[int] = 256, normalized: Optional[bool] = False):
        super(Colormap, self).__init__()
        self.n = n
        self.normalized = normalized

    @staticmethod
    def get_bit_at_idx(val, idx):
        return (val & (1 << idx)) != 0

    def get_color_map(self) -> np.ndarray:

        dtype = "float32" if self.normalized else "uint8"
        color_map = np.zeros((self.n, 3), dtype=dtype)
        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.get_bit_at_idx(c, 0) << 7 - j)
                g = g | (self.get_bit_at_idx(c, 1) << 7 - j)
                b = b | (self.get_bit_at_idx(c, 2) << 7 - j)
                c = c >> 3

            color_map[i] = np.array([r, g, b])
        color_map = color_map / 255 if self.normalized else color_map
        return color_map

    def get_box_color_codes(self) -> List:
        box_codes = []

        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.get_bit_at_idx(c, 0) << 7 - j)
                g = g | (self.get_bit_at_idx(c, 1) << 7 - j)
                b = b | (self.get_bit_at_idx(c, 2) << 7 - j)
                c = c >> 3
            box_codes.append((int(r), int(g), int(b)))
        return box_codes

    def get_color_map_list(self) -> List:
        cmap = self.get_color_map()
        cmap = np.asarray(cmap).flatten()
        return list(cmap)


def draw_bounding_boxes(
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        phenotypes: np.ndarray,
        color_map: Optional = None,  # type: ignore
        object_names: Optional[List] = None,
        is_bgr_format: Optional[bool] = False,
        save_path: Optional[str] = None,
        num_classes: Optional[int] = 81,
        conf_score_threshold=0.5,
) -> None:
    FONT_SIZE = cv2.FONT_HERSHEY_PLAIN
    LABEL_COLOR = [255, 255, 255]
    TEXT_THICKNESS = 1
    RECT_BORDER_THICKNESS = 2

    """Utility function to draw bounding boxes of objects along with their labels and score on a given image"""

    boxes = boxes.astype(np.int32)

    if is_bgr_format:
        # convert from BGR to RGB colorspace
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if color_map is None:
        color_map = Colormap().get_box_color_codes()

    for label, score, coords, pheno in zip(labels, scores, boxes, phenotypes):
        if score < conf_score_threshold:
            continue

        r, g, b = color_map[label]
        c1 = (coords[0], coords[1])
        c2 = (coords[2], coords[3])

        fw, h = pheno

        cv2.rectangle(image, c1, c2, (r, g, b), thickness=RECT_BORDER_THICKNESS)
        if object_names is not None:
            label_text = "{label}: {score:.2f}, fw: {fw:.2f}, h: {h:.2f}".format(
                label=object_names[label], score=score, fw=fw, h=h
            )
            t_size = cv2.getTextSize(label_text, FONT_SIZE, 1, TEXT_THICKNESS)[0]
            new_c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

            cv2.rectangle(image, c1, new_c2, (r, g, b), -1)
            cv2.putText(
                image,
                label_text,
                (c1[0], c1[1] + t_size[1] + 4),
                FONT_SIZE,
                1,
                LABEL_COLOR,
                TEXT_THICKNESS,
            )

    if save_path is not None:
        if cv2.imwrite(save_path, image):
            print("Detection results stored at: {}".format(save_path))
        else:
            print("Failed to store detection results")
    return image


def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    # [0, 1] --> [0, 255]
    img_tensor = torch.mul(img_tensor, 255.0)
    # BCHW --> BHWC
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    img_np = img_tensor.byte().cpu().numpy()
    return img_np


def predict_and_save(
        input_tensor: torch.Tensor,
        model: nn.Module,
        input_np: Optional[np.ndarray] = None,
        device: Optional = torch.device("cpu"),  # type: ignore
        is_coco_evaluation: Optional[bool] = False,
        file_name: Optional[str] = None,
        output_stride: Optional[int] = 32,
        orig_h: Optional[int] = None,
        orig_w: Optional[int] = None,
        *args,
        **kwargs
):
    if input_np is None and not is_coco_evaluation:
        input_np = to_numpy(input_tensor).squeeze(  # convert to numpy
            0
        )  # remove batch dimension

    curr_height, curr_width = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_height // output_stride) * output_stride
    new_w = (curr_width // output_stride) * output_stride

    if new_h != curr_height or new_w != curr_width:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(
            input=input_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    # move data to device
    input_tensor = input_tensor.to(device)

    # prediction
    # We dot scale inside the prediction function because we resize the input tensor such
    # that the dimensions are divisible by output stride.
    prediction: DetectionPredTuple = model.predict(input_tensor, is_scaling=False)

    if orig_w is None:
        assert orig_h is None
        orig_h, orig_w = input_np.shape[:2]
    elif orig_h is None:
        assert orig_w is None
        orig_h, orig_w = input_np.shape[:2]
    assert orig_h is not None and orig_w is not None

    # convert tensors to numpy
    boxes = prediction.boxes.cpu().numpy()
    labels = prediction.labels.cpu().numpy()
    scores = prediction.scores.cpu().numpy()
    phenotypes = prediction.phenotypes.cpu().numpy()

    boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2] * orig_w)
    boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2] * orig_h)

    if is_coco_evaluation:
        return boxes, labels, scores, phenotypes

    detection_res_file_name = None
    if file_name is not None:
        file_name = os.path.basename(file_name)
        res_dir = "detection_results"
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        detection_res_file_name = "{}/{}".format(res_dir, file_name)

    draw_bounding_boxes(
        image=input_np,
        boxes=boxes,
        labels=labels,
        scores=scores,
        phenotypes=phenotypes,
        # some models may not use background class which is present in class names.
        # adjust the class names
        object_names=coco_names,
        is_bgr_format=True,
        save_path=detection_res_file_name,
    )


def read_and_process_image(image_fname: str, *args, **kwargs):
    input_img = Image.open(image_fname).convert("RGB")
    input_np = np.array(input_img)
    orig_w, orig_h = input_img.size

    # Resize the image to the resolution that detector supports
    res_h, res_w = (320, 320)
    input_img = F_vision.resize(
        input_img,
        size=[res_h, res_w],
        interpolation=F_vision.InterpolationMode.BILINEAR,
    )
    input_tensor = F_vision.pil_to_tensor(input_img)
    input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
    return input_tensor, input_np, orig_h, orig_w


def predict_image(image_fname, **kwargs):
    if not os.path.isfile(image_fname):
        print("Image file does not exist at: {}".format(image_fname))

    input_tensor, input_imp_copy, orig_h, orig_w = read_and_process_image(
        image_fname=image_fname
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: SingleShotMaskDetector = torch.load("models/coco-ssd-mobilevitv2-0.75_pretrained.pt", weights_only=False)
    model = model.to(device=device)

    if model.training:
        print("Model is in training mode. Switching to evaluation mode")
        model.eval()

    with torch.no_grad():
        predict_and_save(
            input_tensor=input_tensor,
            input_np=input_imp_copy,
            file_name=image_fname,
            model=model,
            device=device,
            orig_h=orig_h,
            orig_w=orig_w,
        )


def predict_image_2(model, device, image_path):
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        detections = model(input_tensor)

    # Load the image using OpenCV for visualization
    image_cv = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Create figure and axis
    _, ax = plt.subplots(1, figsize=(9, 6))
    ax.imshow(image_cv)

    for detection_pred in detections:
        # Extract the bounding boxes and labels
        boxes = detection_pred["boxes"].cpu().numpy()
        scores = detection_pred["scores"].cpu().numpy()
        labels = detection_pred["labels"].cpu().numpy()
        phenotypes = detection_pred["phenotypes"].cpu().numpy()

        for box, score, label, phenotype in zip(boxes, scores, labels, phenotypes):
            if score < 0.5:
                continue

            x_min, y_min, x_max, y_max = box

            # Width and height of box
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # Class name and confidence
            class_name = coco_names[int(label)] if not args.lettuce_model else lettuce_names[int(label)]
            text = f"{class_name}: {score:.2f}\nfw: {phenotype[0]:.2f}\nh: {phenotype[1]:.2f}"
            plt.text(
                x_min, y_min - 5, text,
                fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.7)
            )

    # Display the image
    plt.axis('off')
    plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    # checkpoint = torch.load("model_95.pth", map_location=torch.device('cpu'), weights_only=False)
    # model.load_state_dict(checkpoint["model"])
    model = LettuceModelEval(image_mean=[0.0, 0.0, 0.0],
                             image_std=[1.0, 1.0, 1.0])
    weights = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    if os.path.isdir(args.image_path):
        files = glob.glob(f"{args.image_path}/*.*")
        for file in files:
            predict_image_2(model, device, file)
    else:
        predict_image_2(model, device, args.image_path)


def main2(args):
    predict_image(args.image_path)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
