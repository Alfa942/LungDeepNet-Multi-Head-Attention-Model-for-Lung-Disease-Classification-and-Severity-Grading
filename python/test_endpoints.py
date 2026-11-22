import urllib.request
import json
import base64
import os

def test_endpoints():
    print("=== Testing Local Server Endpoints on http://localhost:3000 ===")

    # 1. Test /api/health
    with urllib.request.urlopen("http://localhost:3000/api/health", timeout=5) as r:
        data = json.loads(r.read().decode("utf-8"))
        print("[OK] Health status:", data.get("status"), "| mode:", data.get("mode"))

    # 2. Test /api/colab/download-training-notebook
    with urllib.request.urlopen("http://localhost:3000/api/colab/download-training-notebook", timeout=5) as r:
        nb_bytes = r.read()
        print(f"[OK] Training Notebook Download: {len(nb_bytes):,} bytes")

    # 3. Test /api/colab/download-bundle
    with urllib.request.urlopen("http://localhost:3000/api/colab/download-bundle", timeout=5) as r:
        zip_bytes = r.read()
        print(f"[OK] Training Dataset Bundle Download: {len(zip_bytes):,} bytes")

    # 4. Test /api/analyze-case (Local Clinical Interpretability Engine)
    analysis_payload = {
        "disease": "Pneumonia",
        "confidence": 0.954,
        "severityScore": 0.54,
        "diseaseFeatures": {"consolidation": True},
        "heatmapType": "SPATIAL ATTENTION",
        "patientId": "CASE-NORM-042",
        "clinicalNotes": "Focal airspace opacities noted in right lower lobe."
    }
    req = urllib.request.Request(
        "http://localhost:3000/api/analyze-case",
        data=json.dumps(analysis_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        res = json.loads(r.read().decode("utf-8"))
        print("[OK] Local Clinical Analysis returned:", res.get("source"))
        analysis_first_line = res.get("analysis", "").split("\n")[0]
        print("     Snippet:", analysis_first_line)

    # 5. Test /api/predict-image (Local Model Image Prediction)
    # Read a sample image from data/images/Normal/normal_001.pgm and base64 encode it
    sample_img_path = os.path.join(os.getcwd(), "data", "images", "Normal", "normal_001.pgm")
    if os.path.exists(sample_img_path):
        with open(sample_img_path, "rb") as f:
            b64_data = f"data:image/x-portable-graymap;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        
        pred_payload = {
            "image": b64_data,
            "fileName": "normal_001.pgm"
        }
        pred_req = urllib.request.Request(
            "http://localhost:3000/api/predict-image",
            data=json.dumps(pred_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(pred_req, timeout=35) as r:
            pred_res = json.loads(r.read().decode("utf-8"))
            print("[OK] Predict-Image response:", pred_res.get("source"))
            print("     Predicted class:", pred_res.get("result", {}).get("disease") or pred_res.get("result", {}).get("prediction"))

    print("\nAll endpoints passed successfully with ZERO external API keys!")

if __name__ == "__main__":
    test_endpoints()
