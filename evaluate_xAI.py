"""
evaluate_xAI.py

Evaluate Explainable AI models on all Voter datasets.

Usage:
    python evaluate_xAI.py
"""

import torch
from pathlib import Path

from ModelFactory import ModelFactory
import utils


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    factory = ModelFactory(device=device)
    batch_size = 64

    # Models to evaluate
    models_to_evaluate = {
        "expv2_resnet20": "./checkpoint/Explainable_ResNet20.pth",
        "expv2_vgg16": "./checkpoint/Explainable_VGG16.pth",
    }

    # Datasets using utils.py functions
    datasets = {
        "OnlyBubbles - Validation": utils.GetVoterValidation(batch_size),
        "Combined - Validation": utils.GetVoterValidationCombined(batch_size),
        "OnlyBubbles - Training": utils.GetVoterTraining(batch_size),
        "Combined - Training": utils.GetVoterTrainingCombined(batch_size),
    }
    
    print("\n" + "=" * 70)
    print("EXPLAINABLE AI MODEL EVALUATION - ALL DATASETS")
    print("=" * 70)
    
    for model_name, ckpt_path in models_to_evaluate.items():
        print(f"\n{'─' * 70}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'─' * 70}")
        
        if not Path(ckpt_path).exists():
            print(f"  ✗ Checkpoint not found!")
            continue
        
        try:
            model = factory.get_model(model_name, str(ckpt_path))
            
            for dataset_name, loader in datasets.items():
                acc = utils.validateD(loader, model, device)
                print(f"  {dataset_name:<30}: {acc:.4f} ({acc * 100:.2f}%)")
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()