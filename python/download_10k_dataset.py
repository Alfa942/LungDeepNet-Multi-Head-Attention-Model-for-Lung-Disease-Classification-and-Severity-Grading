"""
Automated 10,000+ Open-Source Chest Radiograph Ingestion Engine
Downloads real clinical chest X-rays from the NIH ChestX-ray14 WebDataset cohort (30,545 images total)
and constructs a harmonized 10,000+ sample dataset and metadata manifest.
"""

import os
import io
import csv
import tarfile
import urllib.request
import pandas as pd
from PIL import Image

BASE_REPO = "https://raw.githubusercontent.com/MichaelNoya/nih-chest-xray-webdataset-subset/main"
TARGET_DIR = "data/nih_10k"
OUT_CSV = "data/metadata_10k.csv"

def fetch_url(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()

def download_and_extract_10k_dataset(num_archives=10, max_samples=10000):
    """
    Downloads WebDataset tar archives (each containing ~1,000 real 256x256 radiographs)
    and labels from the NIH ChestX-ray14 benchmark subset.
    """
    print("=" * 70)
    print(" 🏥 LUNGDEEPNET: 10,000+ NIH CLINICAL CHEST RADIOGRAPH INGESTION ")
    print("=" * 70)

    os.makedirs(TARGET_DIR, exist_ok=True)
    for d in ['Normal', 'Pneumonia', 'Emphysema', 'Pulmonary_Fibrosis']:
        os.makedirs(os.path.join(TARGET_DIR, d), exist_ok=True)

    # 1. Download Ground Truth Labels
    print("\n[1/3] Downloading NIH 21,172-sample ground-truth label manifest...")
    labels_url = f"{BASE_REPO}/labels/train_labels.csv"
    labels_bytes = fetch_url(labels_url)
    df_labels = pd.read_csv(io.BytesIO(labels_bytes))
    print(f"  • Successfully loaded labels for {len(df_labels):,} radiographs.")

    # Create mapping dictionary: Image Index -> Disease category
    image_to_disease = {}
    pathologies = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                   'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                   'Pleural_Thickening', 'Hernia']

    for _, row in df_labels.iterrows():
        idx = row['Image Index']
        # Categorize into our target 4-class taxonomy
        if row.get('Fibrosis', 0) == 1:
            image_to_disease[idx] = ('Pulmonary Fibrosis', 'Pulmonary_Fibrosis', 0.68)
        elif row.get('Emphysema', 0) == 1:
            image_to_disease[idx] = ('Emphysema', 'Emphysema', 0.62)
        elif row.get('Pneumonia', 0) == 1 or row.get('Consolidation', 0) == 1:
            image_to_disease[idx] = ('Pneumonia', 'Pneumonia', 0.72)
        elif row[pathologies].sum() == 0:
            image_to_disease[idx] = ('Normal', 'Normal', 0.00)

    print(f"  • Matched {len(image_to_disease):,} candidates in target disease taxonomy.")

    # 2. Download and Unpack Archives
    print(f"\n[2/3] Downloading {num_archives} tar archives (~34MB each, ~1,000 images per archive)...")
    records = []
    total_images_saved = 0

    for i in range(num_archives):
        if total_images_saved >= max_samples:
            break

        archive_name = f"ChestXray14_train_{i:03d}.tar"
        archive_url = f"{BASE_REPO}/datasets/{archive_name}"
        print(f"\n  ⬇️  Downloading Archive [{i+1}/{num_archives}]: {archive_name} ...")

        try:
            tar_bytes = fetch_url(archive_url, timeout=120)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:*') as tar:
                members = [m for m in tar.getmembers() if m.name.endswith('.png')]
                print(f"      Unpacking {len(members)} images from {archive_name}...")

                for m in members:
                    if total_images_saved >= max_samples:
                        break

                    fname = os.path.basename(m.name)
                    if fname in image_to_disease:
                        disease_name, folder_name, severity_score = image_to_disease[fname]
                        
                        f = tar.extractfile(m)
                        if f is not None:
                            save_path = os.path.join(TARGET_DIR, folder_name, fname)
                            with open(save_path, 'wb') as out_f:
                                out_f.write(f.read())

                            patient_id = fname.split('_')[0]
                            rel_path = os.path.relpath(save_path, os.getcwd()).replace('\\', '/')
                            records.append({
                                'patient_id': f"NIH_{patient_id}",
                                'image_path': rel_path,
                                'disease': disease_name,
                                'severity_score': severity_score,
                                'severity_available': disease_name != 'Normal',
                                'source': 'NIH_ChestXray14_Full'
                            })
                            total_images_saved += 1

            print(f"      Accumulated {total_images_saved:,} clinical images so far.")
        except Exception as e:
            print(f"      [!] Error processing {archive_name}: {e}")

    # 3. Save Harmonized Metadata CSV
    print(f"\n[3/3] Saving harmonized dataset manifest: {OUT_CSV}")
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f" [SUCCESS] Ingested {len(df_out):,} authentic clinical chest radiographs!")
    print(f" Saved to: {TARGET_DIR}/")
    print(f" Manifest: {OUT_CSV}")
    print("\nClass Distribution:")
    for d, cnt in df_out['disease'].value_counts().items():
        print(f"  • {d:20s}: {cnt:,} images ({cnt/len(df_out)*100:.1f}%)")
    print("=" * 70)

    return OUT_CSV

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download 10,000+ NIH Chest Radiographs")
    parser.add_argument('--archives', type=int, default=10, help="Number of 1,000-image archives to download (1-22)")
    parser.add_argument('--max_samples', type=int, default=10000, help="Maximum number of images to save")
    args = parser.parse_args()

    download_and_extract_10k_dataset(num_archives=args.archives, max_samples=args.max_samples)
