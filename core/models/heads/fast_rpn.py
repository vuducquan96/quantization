import torch
from torch import nn
import numpy as np
import torch
from core.models.registry import HEAD
from pdb import set_trace as bp

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


@HEAD.register_module()
class FastRPN(nn.Module):
    def __init__(self,
                 config):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """

        super(FastRPN, self).__init__()
        # import pdb; pdb.set_trace()
        self._box_code_size = 6
        self._num_anchor_per_loc = 1
        self._num_input_features = config["num_input_features"]
        self._num_class = 3

        self.conv_cls = nn.Conv2d(self._num_input_features, self._num_class, 1)
        self.conv_box = nn.Conv2d(self._num_input_features,
                                  self._num_anchor_per_loc * self._box_code_size, 1)

    def forward(self, x):

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }

        # return cls_preds, box_preds
        return ret_dict
