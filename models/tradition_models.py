import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------- NAFNet --------------------
class NAFNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_blocks=16, width=32):
        """
        NAFNet 网络结构
        :param in_channels: 输入通道数（默认为1，灰度图像）
        :param out_channels: 输出通道数（默认为1，灰度图像）
        :param num_blocks: 块的数量
        :param width: 每层的通道数
        """
        super(NAFNet, self).__init__()

        # 初始卷积
        self.head = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)

        # 中间块
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks)
        ])

        # 最终卷积
        self.tail = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 初始卷积
        x = self.head(x)

        # 中间块
        for block in self.body:
            x = block(x)

        # 最终卷积
        x = self.tail(x)

        return x
# -------------------- DRUNet --------------------
class DRUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_width=64):
        """
        DRUNet 网络结构
        :param in_channels: 输入通道数（默认为1，灰度图像）
        :param out_channels: 输出通道数（默认为1，灰度图像）
        :param base_width: 基础通道数
        """
        super(DRUNet, self).__init__()

        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.decoder3 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # 解码器
        x = self.decoder3(x3) + x2
        x = self.decoder2(x) + x1
        x = self.decoder1(x)

        return x
# -------------------- MPRNet --------------------
class MPRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_width=32):
        """
        MPRNet 网络结构
        :param in_channels: 输入通道数（默认为1，灰度图像）
        :param out_channels: 输出通道数（默认为1，灰度图像）
        :param base_width: 基础通道数
        """
        super(MPRNet, self).__init__()

        # 第一阶段
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 第二阶段
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 第三阶段
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 上采样和融合
        self.up2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )

        # 最终卷积
        self.tail = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 第一阶段
        x1 = self.stage1(x)

        # 第二阶段
        x2 = self.stage2(x1)

        # 第三阶段
        x3 = self.stage3(x2)

        # 上采样和融合
        x = self.up2(x3) + x2
        x = self.up1(x) + x1

        # 最终卷积
        x = self.tail(x)

        return x
# -------------------- DANet --------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return torch.sigmoid(y)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return torch.sigmoid(y)
class DANet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_width=128):
        """
        DANet 网络结构
        :param in_channels: 输入通道数（默认为1，灰度图像）
        :param out_channels: 输出通道数（默认为1，灰度图像）
        :param base_width: 基础通道数
        """
        super(DANet, self).__init__()

        # 初始卷积
        self.head = nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1)

        # 注意力模块
        self.ca = ChannelAttention(base_width)
        self.sa = SpatialAttention()

        # 中间卷积
        self.body = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 最终卷积
        self.tail = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 初始卷积
        x = self.head(x)

        # 通道注意力
        ca = self.ca(x)
        x = x * ca

        # 空间注意力
        sa = self.sa(x)
        x = x * sa

        # 中间卷积
        x = self.body(x)

        # 最终卷积
        x = self.tail(x)

        return x

# -------------------- 测试代码 --------------------
if __name__ == "__main__":
    # 输入参数
    batch_size = 1
    in_channels = 1
    height, width = 800, 800

    # 随机生成输入图像
    input_image = torch.randn(batch_size, in_channels, height, width)

    # 测试 NAFNet
    nafnet = NAFNet(in_channels=in_channels, out_channels=in_channels)
    output_nafnet = nafnet(input_image)
    print("NAFNet 输出形状:", output_nafnet.shape)

    # 测试 DRUNet
    drunet = DRUNet(in_channels=in_channels, out_channels=in_channels)
    output_drunet = drunet(input_image)
    print("DRUNet 输出形状:", output_drunet.shape)

    # 测试 MPRNet
    mprnet = MPRNet(in_channels=in_channels, out_channels=in_channels)
    output_mprnet = mprnet(input_image)
    print("MPRNet 输出形状:", output_mprnet.shape)

    # 测试 DANet
    danet = DANet(in_channels=in_channels, out_channels=in_channels)
    output_danet = danet(input_image)
    print("DANet 输出形状:", output_danet.shape)