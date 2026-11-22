import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import predict_image

base = r"e:\personal\alfa_project\LungDeepNet-Multi-Head-Attention-Model-for-Lung-Disease-Classification-and-Severity-Grading\data\dataset_40k"
cats = ["Normal", "Pneumonia", "Pulmonary_Fibrosis", "Emphysema"]

for c in cats:
    print(f"\n*** Testing Category: {c} ***")
    for p in glob.glob(os.path.join(base, c, "*.png"))[:5]:
        fname = os.path.basename(p)
        res = predict_image(p)["result"]
        probs = res["posteriorProbabilities"]
        print(f"[{fname}] Predicted: {res['disease']} ({res['confidence']*100:.1f}%) | N={probs['Normal']*100:.1f}%, P={probs['Pneumonia']*100:.1f}%, F={probs['Pulmonary Fibrosis']*100:.1f}%, E={probs['Emphysema']*100:.1f}%")
