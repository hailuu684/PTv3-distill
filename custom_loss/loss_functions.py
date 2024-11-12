import torch.nn as nn
from cross_entropy import CrossEntropyLoss
from lovasz import LovaszLoss

"""
criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    
"""


class NuSceneLoss(nn.Module):
    def __init__(self):
        super().__init__()

        print("-----> Building CrossEntropyLoss")
        self.cross_entropy_loss = CrossEntropyLoss(loss_weight=1.0, ignore_index=-1)

        print("-----> Building LovaszLoss loss")
        self.lovasz_loss = LovaszLoss(mode="multiclass", loss_weight=1.0, ignore_index=-1)

    def forward(self, pred, target):

        loss = self.cross_entropy_loss(pred, target) + self.lovasz_loss(pred, target)

        return loss



