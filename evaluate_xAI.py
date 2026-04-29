# evaluate_xAI.py
"""
Evaluate Explainable AI models on Voter datasets.
"""

import torch
from pathlib import Path

from ModelFactory import ModelFactory
from constants import CHECKPOINTS, DATASETS, EXPERIMENTS_XAI
import utils


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    factory = ModelFactory(device=device)

    # Get XAI models from constants
    models_to_evaluate = {
        "expv2_resnet20": CHECKPOINTS["expv2_resnet20"],
        "expv2_vgg16": CHECKPOINTS["expv2_vgg16"],
    }

    batch_size = 64
    
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
            
            # Evaluate on all datasets from constants
            for dataset_name, dataset_path in DATASETS.items():
                if not Path(dataset_path).exists():
                    print(f"  {dataset_name}: Dataset not found")
                    continue
                
                # Use make_loader from utils.py
                loader = utils.make_loader(str(dataset_path), batch_size, device)
                
                # Use validateD from utils.py
                acc = utils.validateD(loader, model, device)
                print(f"  {dataset_name:20s}: {acc:.4f} ({acc * 100:.2f}%)")
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()