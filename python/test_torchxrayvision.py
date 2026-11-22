import sys
import os
import glob
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, r"e:\personal\alfa_project\torchxrayvision")
import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])

base_cohort = r"e:\personal\alfa_project\LungDeepNet-Multi-Head-Attention-Model-for-Lung-Disease-Classification-and-Severity-Grading\data\dataset_40k"
categories = ["Normal", "Pneumonia", "Pulmonary_Fibrosis", "Emphysema"]

for cat in categories:
    print(f"\n==================== {cat} ====================")
    flist = glob.glob(os.path.join(base_cohort, cat, "*.png"))[:4]
    for p in flist:
        fname = os.path.basename(p)
        pil = Image.open(p).convert("L")
        arr = np.array(pil)
        img = xrv.datasets.normalize(arr, 255)
        img = img[None, ...]
        img = transform(img)
        t = torch.from_numpy(img).unsqueeze(0).float()
        with torch.no_grad():
            preds = model(t)[0].numpy()
        d = dict(zip(model.pathologies, preds))
        
        pneu = float(d.get("Pneumonia", 0))
        fib = float(d.get("Fibrosis", 0))
        emp = float(d.get("Emphysema", 0))
        infil = float(d.get("Infiltration", 0))
        cons = float(d.get("Consolidation", 0))
        eff = float(d.get("Effusion", 0))
        
        print(f"[{fname}] Pneu={pneu:.3f} | Cons={cons:.3f} | Infil={infil:.3f} | Fib={fib:.3f} | Emp={emp:.3f} | Eff={eff:.3f}")
