# check_prototype_params.py

import torch
import sys
from pathlib import Path

base_dir = Path(".")
sys.path.insert(0, str(base_dir / "cosine-is-almost" / "protopnext"))
sys.path.insert(0, str(base_dir))

print("=" * 70)
print("PROTOTYPE LAYER PARAMETERS")
print("=" * 70)

for model_name in ["Explainable_ResNet20.pth", "Explainable_VGG16.pth"]:
    model = torch.load(f"./checkpoint/{model_name}", map_location="cpu", weights_only=False)
    
    print(f"\n{'─' * 70}")
    print(f"Model: {model_name}")
    print(f"{'─' * 70}")
    
    # Prototype layer parameters
    proto_layer = model.prototype_layer
    proto_params = sum(p.numel() for p in proto_layer.parameters())
    proto_trainable = sum(p.numel() for p in proto_layer.parameters() if p.requires_grad)
    
    print(f"\nPrototype Layer:")
    print(f"  Total parameters:     {proto_params:,}")
    print(f"  Trainable parameters: {proto_trainable:,}")
    
    # Prototype tensor shape
    proto_tensors = proto_layer.prototype_tensors
    print(f"\nPrototype Tensors:")
    print(f"  Shape: {proto_tensors.shape}")
    print(f"  • Num prototypes:     {proto_tensors.shape[0]}")
    print(f"  • Channels:           {proto_tensors.shape[1]}")
    print(f"  • Height:             {proto_tensors.shape[2]}")
    print(f"  • Width:              {proto_tensors.shape[3]}")
    print(f"  • Total elements:     {proto_tensors.numel():,}")
    
    # Breakdown of all model components
    print(f"\n{'─' * 40}")
    print("Full Model Breakdown:")
    print(f"{'─' * 40}")
    
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    addon_params = sum(p.numel() for p in model.add_on_layers.parameters())
    pred_head_params = sum(p.numel() for p in model.prototype_prediction_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Backbone:             {backbone_params:>12,}")
    print(f"  Add-on layers:        {addon_params:>12,}")
    print(f"  Prototype layer:      {proto_params:>12,}")
    print(f"  Prediction head:      {pred_head_params:>12,}")
    print(f"  {'─' * 28}")
    print(f"  TOTAL:                {total_params:>12,}")
    
    # Verify sum
    component_sum = backbone_params + addon_params + proto_params + pred_head_params
    if component_sum != total_params:
        print(f"  (Note: component sum = {component_sum:,}, diff = {total_params - component_sum:,})")