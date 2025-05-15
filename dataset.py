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


class LettuceDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.target_keys = {"image_id", "boxes", "labels"}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

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

        for target_key in self.target_keys - {"image_id", "boxes", "masks", "labels"}:
            target[target_key] = batched_target[target_key]

        return image, target

    def get_height_and_width(self, index: int) -> Tuple[int, int]:
        img_info = self.coco.imgs[self.ids[index]]
        return img_info["height"], img_info["width"]

def get_lettuce_data(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    anno_file_template = "{}_{}.json"
    PATHS = {
        "train": ("train", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = LettuceDataset(img_folder, ann_file, transforms=transforms)

    dataset = _coco_remove_images_without_annotations(dataset)

    return dataset
