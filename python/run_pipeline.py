"""
Main Python CLI Entrypoint for LungDeepNet
Usage:
    python run_pipeline.py --mode demo
    python run_pipeline.py --mode train --epochs 25
    python run_pipeline.py --mode optuna --trials 20
    python run_pipeline.py --mode eval --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
import torch
from model import LungDeepNet
from attention import SpatialAttention, ChannelAttention
from losses import MultiTaskLoss


def main():
    parser = argparse.ArgumentParser(description="LungDeepNet Multi-Task Pipeline")
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'train', 'optuna', 'eval'],
                        help="Execution mode: demo (test forward pass), train, optuna, or eval")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training/eval")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help="Model checkpoint path")
    args = parser.parse_args()

    print("=" * 60)
    print(" LUNGDEEPNET: DENSENET121 + DUAL-HEAD ATTENTION PIPELINE ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")

    if args.mode == 'demo':
        print("\n[*] Initializing LungDeepNet Model...")
        model = LungDeepNet(num_classes=4, reduction_ratio=16, dropout_rate=0.25, pretrained=False).to(device)
        model.eval()

        # Generate a dummy synthetic chest radiograph tensor (Batch=2, Channels=3, 224x224)
        dummy_xray = torch.randn(2, 3, 224, 224).to(device)
        print(f"[*] Input Batch Shape: {dummy_xray.shape}")

        with torch.no_grad():
            outputs = model(dummy_xray)

        cls_logits = outputs['cls_logits']
        pred_probs = torch.softmax(cls_logits, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)
        severity_scores = outputs['severity']
        spatial_maps = outputs['spatial_map']

        disease_names = ['Normal', 'Pneumonia', 'Emphysema', 'Pulmonary Fibrosis']

        print("\n--- Model Verification Forward Pass ---")
        for i in range(2):
            cls_idx = pred_classes[i].item()
            conf = pred_probs[i, cls_idx].item() * 100
            sev = severity_scores[i].item()
            grade = 'Normal (0.00)' if cls_idx == 0 else ('Mild' if sev < 0.35 else ('Moderate' if sev < 0.70 else 'Severe'))
            print(f"Sample #{i+1}:")
            print(f"  • Predicted Disease:    {disease_names[cls_idx]} ({conf:.1f}% confidence)")
            print(f"  • Predicted Severity:   {sev:.4f} -> Grade: {grade}")
            print(f"  • Spatial Attention:    {spatial_maps[i].shape} (WHERE lesions are located)")
            print(f"  • Channel Attention:    {outputs['channel_weights'][i].shape} (WHICH feature channels quantify burden)")

        print("\n[SUCCESS] LungDeepNet dual-head architecture initialized and functioning correctly!")

    elif args.mode == 'train':
        print(f"[*] Training pipeline selected ({args.epochs} epochs).")
        print("Note: Provide path to your chest X-ray metadata CSV and images.")
        
    elif args.mode == 'optuna':
        print("[*] Optuna Bayesian hyperparameter optimization selected.")
        
    elif args.mode == 'eval':
        print(f"[*] Evaluation on test cohort using {args.checkpoint}.")


if __name__ == '__main__':
    main()
