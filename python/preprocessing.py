"""
Preprocessing & Medical-Safe Data Augmentations
Uses Albumentations with ImageNet normalization constants for DenseNet121.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Augmentation pipeline for training phase only.
    Strictly medically safe: small rotations, slight scale, modest contrast/brightness.
    No vertical flips (anatomically invalid for erect chest radiographs).
    """
    return A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.06, rotate_limit=7, p=0.5, border_mode=0),
        A.HorizontalFlip(p=0.5), # Standard medically acceptable orientation variance
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transforms():
    """Deterministic pipeline for validation and unseen test evaluation."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
