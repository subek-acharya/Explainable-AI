"""
constants.py

Configuration file containing paths and settings for model experiments.

This file defines:
    - CHECKPOINTS: Paths to pre-trained model checkpoint files (.th, .pth)
                   Supported models: ResNet20, VGG16, CaiT, SVM, SNN, XAI
"""

CHECKPOINTS = {
    # Explainable AI models (XAI)
    "expv2_resnet20": "./checkpoint/Explainable_ResNet20.pth",
    "expv2_vgg16": "./checkpoint/Explainable_VGG16.pth",
}