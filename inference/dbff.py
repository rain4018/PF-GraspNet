import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(x_cat))
        return spatial_att


class DBFF(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(DBFF, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out, _ = self.forward_with_attention(x)
        return x_out

    def forward_with_attention(self, x):
        ca = self.channel_att(x)
        x_ca = x * ca
        sa = self.spatial_att(x_ca)
        x_out = x_ca * sa
        return x_out, sa