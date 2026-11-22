"""
Dataset Harmonization & Patient-Wise DataLoaders
Prevents patient leakage across train/val/test splits.
Supports DICOM (.dcm), PNG, and JPEG.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


DISEASE_MAP = {
    'Normal': 0,
    'Pneumonia': 1,
    'Emphysema': 2,
    'Pulmonary Fibrosis': 3
}
INV_DISEASE_MAP = {v: k for k, v in DISEASE_MAP.items()}


class ChestXRayMultiTaskDataset(Dataset):
    """
    Harmonized multi-task chest X-ray dataset.
    Loads standardized DICOM / PNG / JPEG images, applies augmentations,
    and returns (image, disease_label, severity_score, severity_mask, patient_id).
    """
    def __init__(self, metadata_df, transform=None, is_train=False):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        disease_name = row['disease']
        disease_label = DISEASE_MAP[disease_name]
        
        # Severity target: continuous score in [0.0, 1.0]
        has_severity = bool(row.get('severity_available', False)) and not pd.isna(row.get('severity_score', np.nan))
        severity_score = float(row.get('severity_score', 0.0)) if has_severity else 0.0
        severity_mask = 1.0 if (has_severity and disease_name != 'Normal') else 0.0
        
        # Load image (supports DICOM, PNG, JPG)
        image = self._load_image(image_path)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return {
            'image': image,
            'disease_label': torch.tensor(disease_label, dtype=torch.long),
            'severity_score': torch.tensor(severity_score, dtype=torch.float32),
            'severity_mask': torch.tensor(severity_mask, dtype=torch.float32),
            'patient_id': str(row.get('patient_id', 'UNKNOWN')),
            'disease_name': disease_name
        }

    def _load_image(self, path):
        if path.endswith('.dcm'):
            try:
                import pydicom
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)
                slope = getattr(dcm, 'RescaleSlope', 1)
                intercept = getattr(dcm, 'RescaleIntercept', 0)
                img = img * slope + intercept
                if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                    img = np.max(img) - img
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * 255.0
                img = img.astype(np.uint8)
            except Exception:
                img = np.zeros((224, 224), dtype=np.uint8)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
                
        # Resize to 224x224 and expand to 3 channels for DenseNet121 ImageNet compatibility
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


def get_patient_wise_splits(metadata_csv_path, train_ratio=0.75, val_ratio=0.125, test_ratio=0.125, seed=42):
    """
    Partitions the dataset grouped strictly by patient_id to prevent data leakage.
    """
    df = pd.read_csv(metadata_csv_path)
    np.random.seed(seed)
    
    unique_patients = df['patient_id'].unique()
    np.random.shuffle(unique_patients)
    
    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    print(f"Patient-Wise Split Completed:")
    print(f"  Train: {len(train_df)} images ({len(train_patients)} patients)")
    print(f"  Val:   {len(val_df)} images ({len(val_patients)} patients)")
    print(f"  Test:  {len(test_df)} images ({len(test_patients)} patients)")
    
    # Assert zero patient overlap
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage between train and val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage between train and test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage between val and test!"
    
    return train_df, val_df, test_df
