"""
Preprocessing & Medical-Safe Data Augmentations
Supports Albumentations when available, with torchvision fallback.
Standardized ImageNet normalization constants for DenseNet121.
"""

from typing import Any, Dict
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TorchvisionTransformWrapper:
    """Wrapper that mimics Albumentations interface: transform(image=img)['image']"""
    def __init__(self, tv_transform):
        self.tv_transform = tv_transform

    def __call__(self, image: Any) -> Dict[str, torch.Tensor]:
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = Image.fromarray(np.array(image, dtype=np.uint8))
        
        tensor = self.tv_transform(pil_img)
        return {'image': tensor}


def get_train_transforms():
    """
    Augmentation pipeline for training phase.
    Strictly medically safe: modest rotations, slight scale, slight brightness/contrast.
    No vertical flips (anatomically invalid for erect chest radiographs).
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(224, 224),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.06, rotate_limit=7, p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        tv = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
            T.ColorJitter(brightness=0.10, contrast=0.10),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return TorchvisionTransformWrapper(tv)


def get_val_transforms():
    """Deterministic pipeline for validation and unseen test evaluation."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        tv = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return TorchvisionTransformWrapper(tv)
