"""
Colab Dataset & Code Bundle Exporter
Zips python/ source modules and data/ folder into lungdeepnet_bundle.zip
for instant one-click upload to Google Colab Free Tier.
"""

import os
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(BASE_DIR, "python")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_ZIP_PATHS = [
    os.path.join(BASE_DIR, "lungdeepnet_bundle.zip"),
    os.path.join(BASE_DIR, "public", "lungdeepnet_bundle.zip")
]


def create_bundle():
    print(f"[*] Packaging LungDeepNet Colab bundle from: {BASE_DIR}")
    
    os.makedirs(os.path.join(BASE_DIR, "public"), exist_ok=True)
    target_zip = OUTPUT_ZIP_PATHS[0]
    
    file_count = 0
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add all python files
        if os.path.exists(PYTHON_DIR):
            for root, _, files in os.walk(PYTHON_DIR):
                for f in files:
                    if f.endswith(('.py', '.txt', '.md')) and not f.startswith('.'):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, BASE_DIR)
                        zipf.write(full_path, rel_path)
                        file_count += 1
                        
        # 2. Add all data files (images + metadata.csv)
        if os.path.exists(DATA_DIR):
            for root, _, files in os.walk(DATA_DIR):
                for f in files:
                    if not f.startswith('.'):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, BASE_DIR)
                        zipf.write(full_path, rel_path)
                        file_count += 1

    # Also copy to public/ so Vite can serve it as a static download
    import shutil
    shutil.copy2(target_zip, OUTPUT_ZIP_PATHS[1])
    
    size_mb = os.path.getsize(target_zip) / (1024 * 1024)
    print(f"[SUCCESS] Packaged {file_count} files into '{target_zip}' ({size_mb:.2f} MB)")
    print(f"[SUCCESS] Staged for web download at '{OUTPUT_ZIP_PATHS[1]}'")
    return target_zip


if __name__ == '__main__':
    create_bundle()
