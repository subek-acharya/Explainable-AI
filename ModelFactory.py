import torch
import torch.nn as nn
from typing import Union, List, Optional
import sys
from pathlib import Path

# ----------- Wrapper Class ---------------------
class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            raise KeyError("Model output dict missing 'logits' key.")
        if isinstance(output, tuple):
            return output[0]
        return output


class ModelFactory:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Base directory (where ModelFactory.py is located)
        self.base_dir = Path(__file__).resolve().parent

    def get_model(
        self,
        model_name: str,
        checkpoint_path: Union[str, List[str]],
    ) -> nn.Module:
        model_name = model_name.lower()

        if "expv2" in model_name or "explainable" in model_name:
            return self._create_ppnet_v2_direct(checkpoint_path)
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")

    def _create_ppnet_v2_direct(self, checkpoint_path: str) -> nn.Module:
        """
        Load Explainable AI (ProtoPNet v2) model.
        
        Explanaible_AI/
        ├── ModelFactory.py       
        ├── cosine-is-almost/
        │   └── protopnext/
        │       └── protopnet/
        └── models/
            └── architecture/
                └── ResNet.py
        """
        
        # Path to cosine-is-almost repo
        _COSINE_DIR = self.base_dir / "cosine-is-almost"
        _PPNEXT_DIR = _COSINE_DIR / "protopnext"
        
        # Path to base directory (for models.architecture.ResNet)
        _BASE_DIR = self.base_dir

        # ═══════════════════════════════════════════════════════════════
        # Add ALL required paths to sys.path
        # ═══════════════════════════════════════════════════════════════
        
        paths_to_add = [
            str(_PPNEXT_DIR),    # For: from protopnet.* import ...
            str(_COSINE_DIR),    # For: other cosine-is-almost imports
            str(_BASE_DIR),      # For: from models.architecture.ResNet import ...
        ]
        
        paths_added = []
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)
                paths_added.append(p)

        try:
            ppnet = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )
            ppnet = ppnet.to(self.device)
            ppnet.eval()
            wrapped = LogitsOnlyWrapper(ppnet).to(self.device)
            wrapped.eval()
            return wrapped

        except Exception as e:
            print(f"Failed to load v2 PPNet from {checkpoint_path}: {e}")
            raise
        finally:
            # Clean up added paths
            for p in paths_added:
                if p in sys.path:
                    sys.path.remove(p)