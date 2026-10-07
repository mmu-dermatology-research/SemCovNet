"""
models/transforms.py
─────────────────────────────────────────────────────────────────────────────
Image transforms for the DINOv3 pipeline.

Contract with data_loaders.BaselineDataset
──────────────────────────────────────────
    image = self.transform(image=image)["image"].astype("float32")
    image = torch.tensor(image.transpose(2, 0, 1)).float()

i.e. the dataset does the HWC -> CHW transpose itself, so these transforms
must end at `A.Normalize(...)` and must NOT include ToTensorV2.

Normalisation
─────────────
DINOv3 was trained with the standard ImageNet statistics; open_clip models
use the OpenAI CLIP statistics. `backbone=` picks the right pair so a single
call site works for both registry variants.

Resolution
──────────
224 is the primary configuration (14x14 = 196 patch tokens). 336 (21x21 =
441) is supported for a resolution-sensitivity study, but the plan is
explicit that a larger resolution must not be used for SemCovNet only.
"""

from __future__ import annotations

from typing import Tuple

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_STATS = {
    "dinov3": (IMAGENET_MEAN, IMAGENET_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "clip": (CLIP_MEAN, CLIP_STD),
}


def _random_resized_crop(image_size: int, scale: Tuple[float, float]):
    """albumentations >=1.4 uses `size=(h, w)`; older releases use height/width."""
    try:
        return A.RandomResizedCrop(size=(image_size, image_size), scale=scale,
                                   ratio=(0.9, 1.111))
    except TypeError:                                   # pragma: no cover
        return A.RandomResizedCrop(height=image_size, width=image_size,
                                   scale=scale, ratio=(0.9, 1.111))


def get_encoder_transforms(image_size: int = 224,
                           backbone: str = "dinov3",
                           hflip: bool = True,
                           scale: Tuple[float, float] = (0.7, 1.0),
                           color_jitter: float = 0.0):
    """
    Returns (transforms_train, transforms_val).
    """
    mean, std = _STATS[backbone.lower()]
    resize = int(round(image_size * 1.14))     # 224 -> 256, 336 -> 384

    train_ops = [_random_resized_crop(image_size, scale)]
    if hflip:
        train_ops.append(A.HorizontalFlip(p=0.5))
    if color_jitter > 0:
        train_ops.append(A.ColorJitter(brightness=color_jitter,
                                       contrast=color_jitter,
                                       saturation=color_jitter,
                                       hue=color_jitter / 3, p=0.5))
    train_ops.append(A.Normalize(mean=mean, std=std))

    val_ops = [
        A.Resize(resize, resize),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=mean, std=std),
    ]
    return A.Compose(train_ops), A.Compose(val_ops)


# Backwards-compatible alias with the previous project's util name.
def get_transforms(image_size: int = 224, backbone: str = "dinov3"):
    return get_encoder_transforms(image_size, backbone)
