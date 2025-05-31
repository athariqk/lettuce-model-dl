from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from PIL import Image

from coco_utils import _coco_remove_images_without_annotations


class LettuceRGBDDataset(VisionDataset):
    def __init__(
            self,
            root: Union[str, Path],
            annFile: str,
            depth_image_suffix: str,
            phenotype_names: list[str],
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
        self.phenotype_names = phenotype_names

    def _load_image(self, id: int) -> Image.Image:
        info = self.coco.loadImgs(id)[0]
        subfolder = "rgb" if "rgb" in info["image_type"] else "depth"
        return Image.open(os.path.join(self.root, subfolder, info["file_name"])).convert("RGB")

    def _load_image_name(self, file_name: str, image_type: str) -> Image.Image:
        subfolder = "rgb" if "rgb" in image_type else "depth"
        return Image.open(os.path.join(self.root, subfolder, file_name)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _load_image_pair(self, id: int) -> Tuple[Image.Image, Optional[Image.Image]]:
        img_info = self.coco.loadImgs(id)[0]
        img = self._load_image_name(img_info["file_name"], img_info["image_type"])

        if "paired_id" in img_info:
            paired_img_info = self.coco.loadImgs(img_info["paired_id"])[0]
            paired_img = self._load_image_name(paired_img_info["file_name"], paired_img_info["image_type"])
        else:
            raise ValueError(f"Could not find paired image for {id}")

        return img, paired_img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image, paired_img = self._load_image_pair(id)
        target = self._load_target(id)

        image, target = self.wrap_to_tv2(index, (image, target))

        image_pair = [image, paired_img]

        if self.transforms:
            image_pair, target = self.transforms(image_pair, target)

        return image_pair, target

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
            all_phenotype_values_for_instances = []

            for attributes in batched_target.get("attributes", []):
                current_instance_phenotypes = []
                for pheno_name in self.phenotype_names:
                    value = attributes.get(pheno_name, 0.0)
                    # fresh weight has log-normal distribution, try to bring down variance
                    if pheno_name == "fresh_weight_ln":
                        value = np.log(value)
                    current_instance_phenotypes.append(value)
                all_phenotype_values_for_instances.append(current_instance_phenotypes)

            if all_phenotype_values_for_instances:
                phenotypes = torch.tensor(
                    all_phenotype_values_for_instances,
                    dtype=torch.float32
                )
                target["phenotypes"] = phenotypes
            elif self.phenotype_names:
                phenotypes = torch.empty(
                    (0, len(self.phenotype_names)),
                    dtype=torch.float32
                )
                target["phenotypes"] = phenotypes

        for target_key in self.target_keys - {"image_id", "boxes", "masks", "labels", "phenotypes"}:
            target[target_key] = batched_target[target_key]

        return image, target

    def get_height_and_width(self, index: int) -> Tuple[int, int]:
        img_info = self.coco.imgs[self.ids[index]]
        return img_info["height"], img_info["width"]


def get_lettuce_data(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False,
                     phenotype_names=None):
    if phenotype_names is None:
        phenotype_names = ["fresh_weight", "height"]

    anno_file_template = "{}_{}.json"
    PATHS = {
        "train": ("train", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = LettuceRGBDDataset(img_folder, ann_file, transforms=transforms, depth_image_suffix=".png", phenotype_names=phenotype_names)

    dataset = _coco_remove_images_without_annotations(dataset)

    return dataset

def get_lettuce_data_no_h(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    return get_lettuce_data(root, image_set, transforms, mode, use_v2, with_masks, ["fresh_weight"])
