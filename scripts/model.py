import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBEDDING_DIM = 128

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)