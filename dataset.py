from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from PIL import Image

from coco_utils import _coco_remove_images_without_annotations


class CocoRGBDDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        depth_image_suffix: str = "",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.target_keys = {"image_id", "boxes", "labels", "phenotypes"}

        self.depth_image_suffix = depth_image_suffix

    def _load_image(self, id: int) -> Image.Image:
        info = self.coco.loadImgs(id)[0]
        subfolder = "rgb"
        return Image.open(os.path.join(self.root, subfolder, info["file_name"])).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _load_image_pair(self, id: int) -> Tuple[Image.Image, Optional[Image.Image]]:
        img_info = self.coco.loadImgs(id)[0]

        img = self._load_image(img_info["file_name"])
        if "paired_id" in img_info:
            paired_img_info = self.coco.loadImgs(id)[img_info["paired_id"]]
            path = os.path.join(self.root, paired_img_info["image_type"], paired_img_info["file_name"])
            aux = Image.open(path).convert("RGB")
        else:
            if "rgb" in img_info["image_type"]:
                # Determine depth image path
                base_name, _ = os.path.splitext(img_info["file_name"])
                numbers = ''.join([char for char in base_name if char.isdigit()])
                depth_file_name_short = f"depth{numbers}{self.depth_image_suffix}"
                depth_path = os.path.join(self.root, "depth", depth_file_name_short)
                aux = Image.open(depth_path).convert("RGB")
            else:
                aux = None

        return img, aux

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        image, target = self.wrap_to_tv2(index, (image, target))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    def list_of_dicts_to_dict_of_lists(self, list_of_dicts):
        dict_of_lists = defaultdict(list)
        for dct in list_of_dicts:
            for key, value in dct.items():
                dict_of_lists[key].append(value)
        return dict(dict_of_lists)

    def wrap_to_tv2(self, idx, sample):
        image_id = self.ids[idx]

        image, target = sample

        if not target:
            return image, dict(image_id=image_id)

        canvas_size = tuple(F.get_size(image))

        batched_target = self.list_of_dicts_to_dict_of_lists(target)
        target = {}

        if "image_id" in self.target_keys:
            target["image_id"] = image_id

        if "boxes" in self.target_keys:
            target["boxes"] = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    batched_target["bbox"],
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            )

        if "labels" in self.target_keys:
            target["labels"] = torch.tensor(batched_target["category_id"])

        if "phenotypes" in self.target_keys:
            fresh_weight_values = []
            height_values = []

            for attributes in batched_target.get("attributes", []):
                fresh_weight_values.append(attributes.get("fresh_weight", 0.0))
                height_values.append(attributes.get("height", 0.0))

            # Stack the values to form a tensor of shape [num_instances, 2]
            # where the first column is fresh_weight and second column is height
            phenotypes = torch.tensor(
                [[fw, h] for fw, h in zip(fresh_weight_values, height_values)],
                dtype=torch.float32
            )

            target["phenotypes"] = phenotypes

        for target_key in self.target_keys - {"image_id", "boxes", "masks", "labels", "phenotypes"}:
            target[target_key] = batched_target[target_key]

        return image, target

    def get_height_and_width(self, index: int) -> Tuple[int, int]:
        img_info = self.coco.imgs[self.ids[index]]
        return img_info["height"], img_info["width"]

def get_rgbd_data(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    anno_file_template = "{}_{}.json"
    PATHS = {
        "train": ("train", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoRGBDDataset(img_folder, ann_file, transforms=transforms, depth_image_suffix=".png")

    dataset = _coco_remove_images_without_annotations(dataset)

    return dataset
