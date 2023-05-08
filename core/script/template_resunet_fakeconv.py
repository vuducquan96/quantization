import torch
import torch.nn as nn
from pdb import set_trace as bp
from maskconv import MaskConv

class ResnetUnetFakeconv(nn.Module):
    def __init__(self, backbone):

        super(ResnetUnetFakeconv, self).__init__()
        weights = torch.load("weights.pth").cpu()
        self.first_layer = MaskConv(weights)
        # bp()
        self.backbone = backbone
        
    def forward(self, indice, mask):
        x = self.first_layer(indice, mask)
        # bp()
        cls_preds, box_preds = self.backbone(x)    
        print("Sum:", torch.sum(cls_preds))
        return cls_preds, box_preds
