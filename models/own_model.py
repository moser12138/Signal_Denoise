import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------  轻量级卷积模块  --------------------- #
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ---------------------  通道注意力模块  --------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---------------------  全局特征增强模块（GFE）  --------------------- #
class GlobalFeatureEnhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + x)


# ---------------------  跨尺度特征聚合（MSFA）  --------------------- #
class MultiScaleFeatureAggregation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=7, padding=3)
        self.conv_fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.conv_fuse(torch.cat([x1, x2, x3], dim=1))


# ---------------------  主网络结构  --------------------- #
class SignalDenoiseNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.input_layer = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.encoder = nn.Sequential(
            GlobalFeatureEnhancement(base_channels),
            ChannelAttention(base_channels),
            MultiScaleFeatureAggregation(base_channels),
        )

        self.decoder = nn.Sequential(
            GlobalFeatureEnhancement(base_channels),
            ChannelAttention(base_channels),
            MultiScaleFeatureAggregation(base_channels),
        )

        self.output_layer = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x


# ---------------------  测试网络  --------------------- #
if __name__ == "__main__":
    model = SignalDenoiseNet()
    input_tensor = torch.randn(1, 1, 128, 128)  # 示例输入：单通道 STFT 频谱图
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
