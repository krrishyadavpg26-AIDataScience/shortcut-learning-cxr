"""ResNet-18 classifier for single-channel chest X-rays."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CXRNet(nn.Module):
    """ImageNet ResNet-18 with a 2-class head.

    Takes raw grayscale images in [0, 1] with shape (N, 1, H, W); channel repetition and
    ImageNet normalisation happen inside forward(), so Grad-CAM sees the raw image.
    """

    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.net(x)

    def cam_layer(self, name="layer3"):
        """Last block of the chosen ResNet stage, used as the Grad-CAM target layer."""
        return getattr(self.net, name)[-1]
