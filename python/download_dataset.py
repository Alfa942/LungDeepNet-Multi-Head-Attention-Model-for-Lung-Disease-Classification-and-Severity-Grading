"""
Dataset Ingestion, Download Guide & Synthetic Sample Generator for LungDeepNet

This script provides:
1. Direct download links and API instructions for clinical benchmark datasets:
   - COVID-19 Radiography Database (Kaggle / Qatar University)
   - NIH ChestX-ray14 (NIH Clinical Center)
   - RSNA Pneumonia Detection Challenge
2. An automated `--create-demo` generator that builds an instant localized dataset structure
   with `metadata.csv` and sample radiograph images for immediate pipeline testing.
"""

import os
import sys
import csv
import math
import random
import argparse

try:
    import numpy as np
    import pandas as pd
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

DATASET_RESOURCES = {
    "COVID-19 Radiography Database": {
        "source": "Kaggle / Qatar University & University of Dhaka",
        "url": "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
        "cli_command": "kaggle datasets download -d tawsifurrahman/covid19-radiography-database",
        "description": "21,165 chest X-rays categorized into Normal, COVID-19, Lung Opacity, and Viral Pneumonia."
    },
    "NIH ChestX-ray14": {
        "source": "NIH Clinical Center (National Institutes of Health)",
        "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "cli_command": "wget https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "description": "112,120 frontal-view X-ray images of 30,805 unique patients with 14 disease labels."
    },
    "RSNA Pneumonia Detection Challenge": {
        "source": "Kaggle / Radiological Society of North America",
        "url": "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data",
        "cli_command": "kaggle competitions download -c rsna-pneumonia-detection-challenge",
        "description": "Contains DICOM format radiographs with lung opacity bounding boxes and opacity severity."
    }
}


def print_dataset_sources():
    print("=" * 70)
    print(" 🏥 LUNGDEEPNET: BENCHMARK MEDICAL DATASETS & ACCESS INFORMATION ")
    print("=" * 70)
    for name, info in DATASET_RESOURCES.items():
        print(f"\n📦 {name}")
        print(f"   Source:      {info['source']}")
        print(f"   URL:         {info['url']}")
        print(f"   CLI Command: {info['cli_command']}")
        print(f"   Overview:    {info['description']}")
    print("\n" + "=" * 70)
    print("Expected local directory structure after download:")
    print("data/")
    print("  ├── metadata.csv  (columns: patient_id, image_path, disease, severity_score, severity_available)")
    print("  └── images/")
    print("        ├── Normal/")
    print("        ├── Pneumonia/")
    print("        ├── Emphysema/")
    print("        └── Pulmonary Fibrosis/")
    print("=" * 70)


def create_demo_dataset(target_dir="data", num_samples_per_class=15):
    """
    Generates a functional mock dataset with realistic anatomical phantom radiographs
    and metadata.csv so you can immediately train and test without downloading 50GB.
    """
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
    classes = ['Normal', 'Pneumonia', 'Emphysema', 'Pulmonary Fibrosis']
    
    rows = []
    patient_counter = 1000
    
    print(f"\n[*] Generating demo clinical radiograph dataset in: '{target_dir}'...")
    
    for disease in classes:
        disease_folder = os.path.join(target_dir, "images", disease.replace(" ", "_"))
        os.makedirs(disease_folder, exist_ok=True)
        
        for i in range(num_samples_per_class):
            patient_id = f"PATIENT_{patient_counter}"
            patient_counter += 1
            filename = f"{disease.lower().replace(' ', '_')}_{i+1:03d}.png"
            filepath = os.path.join(disease_folder, filename)
            
            if HAS_CV2:
                # Synthesize synthetic thoracic phantom with lung fields and opacity textures
                canvas = np.zeros((224, 224), dtype=np.uint8)
                cv2.ellipse(canvas, (112, 112), (90, 100), 0, 0, 360, 180, 2)
                cv2.ellipse(canvas, (75, 110), (32, 65), 0, 0, 360, 110, -1)
                cv2.ellipse(canvas, (149, 110), (32, 65), 0, 0, 360, 110, -1)
                cv2.ellipse(canvas, (120, 140), (28, 22), -30, 0, 360, 160, -1)
                
                if disease == 'Normal':
                    severity = 0.0
                    has_severity = False
                elif disease == 'Pneumonia':
                    severity = float(np.clip(np.random.uniform(0.20, 0.85), 0.0, 1.0))
                    has_severity = True
                    cv2.circle(canvas, (150, 135), int(12 + 15 * severity), 200, -1)
                elif disease == 'Emphysema':
                    severity = float(np.clip(np.random.uniform(0.15, 0.75), 0.0, 1.0))
                    has_severity = True
                    cv2.ellipse(canvas, (75, 110), (36, 75), 0, 0, 360, 60, -1)
                    cv2.ellipse(canvas, (149, 110), (36, 75), 0, 0, 360, 60, -1)
                else: # Pulmonary Fibrosis
                    severity = float(np.clip(np.random.uniform(0.30, 0.95), 0.0, 1.0))
                    has_severity = True
                    for y in range(80, 160, 10):
                        cv2.line(canvas, (50, y), (90, y + 5), 180, 1)
                        cv2.line(canvas, (135, y), (175, y + 5), 180, 1)
                
                noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
                canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                cv2.imwrite(filepath, canvas)
            else:
                # Pure Python Fallback: Write valid PGM (portable graymap) image binary
                # OpenCV and PyTorch torchvision can read PGM images transparently
                pgm_path = filepath.replace(".png", ".pgm")
                filepath = pgm_path
                if disease == 'Normal':
                    severity = 0.0
                    has_severity = False
                elif disease == 'Pneumonia':
                    severity = round(random.uniform(0.20, 0.85), 3)
                    has_severity = True
                elif disease == 'Emphysema':
                    severity = round(random.uniform(0.15, 0.75), 3)
                    has_severity = True
                else:
                    severity = round(random.uniform(0.30, 0.95), 3)
                    has_severity = True
                
                # Write minimal P5 binary PGM image of 224x224
                with open(filepath, 'wb') as f:
                    f.write(f"P5\n224 224\n255\n".encode('ascii'))
                    # Generate simple synthetic lung outline bytes
                    pixels = bytearray(224 * 224)
                    for py in range(224):
                        for px in range(224):
                            # Distance to left and right lung center
                            d_left = math.sqrt((px - 75)**2 + ((py - 110) * 0.7)**2)
                            d_right = math.sqrt((px - 149)**2 + ((py - 110) * 0.7)**2)
                            if d_left < 35 or d_right < 35:
                                val = 90
                                if disease == 'Pneumonia' and math.sqrt((px - 150)**2 + (py - 135)**2) < 20:
                                    val = 200
                                pixels[py * 224 + px] = val
                            else:
                                pixels[py * 224 + px] = 150 if math.sqrt((px - 112)**2 + ((py - 112)*0.8)**2) < 95 else 20
                    f.write(pixels)
            
            rows.append({
                'patient_id': patient_id,
                'image_path': filepath,
                'disease': disease,
                'severity_score': round(severity, 3),
                'severity_available': has_severity
            })
            
    csv_path = os.path.join(target_dir, "metadata.csv")
    fieldnames = ['patient_id', 'image_path', 'disease', 'severity_score', 'severity_available']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[✓] Created demo dataset with {len(rows)} images across 4 disease classes.")
    print(f"[✓] Metadata CSV generated at: {csv_path}")
    print(f"\nYou can now run training immediately:")
    print(f"   python python/run_pipeline.py --mode train --epochs 5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LungDeepNet Dataset Ingestion Utility")
    parser.add_argument("--create-demo", action="store_true", help="Generate a local mock dataset with 60 images for instant testing")
    parser.add_argument("--dir", type=str, default="data", help="Output directory for data (default: 'data')")
    args = parser.parse_args()

    print_dataset_sources()

    if args.create_demo:
        create_demo_dataset(target_dir=args.dir)
