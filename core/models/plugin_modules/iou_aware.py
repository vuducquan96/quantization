import torch.nn as nn


class IOU_AWARE(nn.Module):
    def __init__(self,inp_conv, out_conv , share_conv_channel=256):

        super(IOU_AWARE, self).__init__()

        self.share_conv_channel = share_conv_channel
        self.iou_shared_conv = nn.Sequential(
             nn.Conv2d(inp_conv, self.share_conv_channel,
             kernel_size=3, padding=1, bias=True),
             nn.BatchNorm2d(self.share_conv_channel),
             nn.ReLU(inplace=True)
        )

        self.block = nn.Sequential(
            nn.Conv2d(self.share_conv_channel, self.share_conv_channel//4, kernel_size=3,stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.share_conv_channel//4),
            nn.ReLU(),
            nn.Conv2d(self.share_conv_channel//4, out_conv, kernel_size=3,stride=1, padding=1, bias=True),
        )


    def forward(self, x):
        x = self.iou_shared_conv(x)
        x = self.block(x)
        return x


