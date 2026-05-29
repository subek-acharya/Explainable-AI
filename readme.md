# Explainable AI (ProtoPNet) for Voter Ballot Classification

Explainable AI models using ProtoPNet architecture for ballot bubble detection in voting systems. These models provide interpretable predictions by learning class-specific prototypes that correspond to real image patches.

## Architecture Overview

We implement ProtoPNet with two backbone architectures adapted for grayscale ballot images (40×50 pixels, single channel):

| Component | ResNet-20 | VGG-16 |
|-----------|-----------|--------|
| Backbone | 567,776 | 14,721,984 |
| Add-on Layers | 131,584 | 525,312 |
| Prototype Layer | 10,240 | 20,480 |
| Prediction Head | 80 | 80 |
| **Total Parameters** | **709,680** | **15,267,856** |

### Model Components

1. **Backbone Network**: Extracts convolutional features from input images. ResNet-20 produces 256-channel feature maps; VGG-16 produces 512-channel feature maps.

2. **Add-on Layers**: 1×1 convolutions with Sigmoid activation that project features into the prototype space.

3. **Prototype Layer**: 40 learned prototypes (20 per class) using cosine similarity:

$$\text{similarity}(z, p) = \frac{z \cdot p}{\|z\| \cdot \|p\|}$$

4. **Prediction Head**: Fully connected layer mapping 40 prototype similarities to 2 class logits (80 parameters).

### Prototype Configuration

- Number of prototypes: 40 (20 per class)
- Prototype tensor shape: `[40, 256, 1, 1]` (ResNet-20) or `[40, 512, 1, 1]` (VGG-16)
- Activation: Cosine similarity
- Prototype sources: Projected to real training patches

## Results

| Model | OnlyBubbles Val | Combined Val | OnlyBubbles Train | Combined Train |
|-------|----------------|--------------|-------------------|----------------|
| xAI-ResNet20 | 1.0000 | 0.9959 | 1.0000 | 0.9996 |
| xAI-VGG16 | 1.0000 | 0.9954 | 1.0000 | 0.9986 |

## Project Structure

```bash
Explanaible_AI/
├── README.md
├── check_prototype_params.py    # Inspect prototype layer parameters
├── evaluate_xAI.py              # Evaluate models on all datasets
├── ModelFactory.py              # Model loading factory
├── utils.py                     # Data loading and validation utilities
├── checkpoint/
│   ├── Explainable_ResNet20.pth # Trained ResNet-20 ProtoPNet
│   └── Explainable_VGG16.pth   # Trained VGG-16 ProtoPNet
├── models/
│   ├── init.py
│   └── architecture/
│       ├── init.py
│       └── ResNet.py            # Custom ResNet-20 architecture
└── cosine-is-almost/            # ProtoPNext framework (submodule)
└── protopnext/
└── protopnet/
```

## Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.5+
- CUDA (for GPU inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/explainable-ai.git
cd explainable-ai

# Clone the ProtoPNext submodule
git clone https://github.com/protopnext/cosine-is-almost.git

# Install dependencies
pip install torch torchvision numpy
```

## Usage

### Evaluate models
```python
python evaluate_xAI.py
```

### Inspect Prototype Parameters
```python
python check_prototype_params.py
```

## References

1. Chen, Chaofan, et al. ["This looks like that: deep learning for interpretable image recognition."](https://arxiv.org/abs/1806.10574) Advances in neural information processing systems 32 (2019) .