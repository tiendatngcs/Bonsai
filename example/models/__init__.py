"""CNN and transformer model definitions."""

from .googlenet import googlenet
from .inceptionv3 import inceptionv3
from .resnet import resnet18, resnet50, resnet152

CNN_MODELS = {
    "googlenet": googlenet,
    "inceptionv3": inceptionv3,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet152": resnet152,
}

__all__ = ["CNN_MODELS"]
