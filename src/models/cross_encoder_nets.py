import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

from core.config_default import DefaultConfig

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块：学习"关注什么特征通道" """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块：学习"关注特征图的哪个空间区域" """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class CBAM(nn.Module):
    """卷积块注意力机制 (Convolutional Block Attention Module)

    顺序应用通道注意力和空间注意力，在不显著增加参数量的情况下
    提纯特征表达，强调对视线估计最有价值的通道和空间区域。
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.total_features = np.sum(list(config.feature_sizes.values()))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


class BaselineEncoder(Encoder):
    def __init__(self):
        super(BaselineEncoder, self).__init__()

        self.cnn_layers = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                                 num_classes=self.total_features,
                                 norm_layer=nn.InstanceNorm2d)

        # CBAM: 在 ResNet-18 Layer4 输出后、全局池化前添加单个 CBAM
        # Layer4 输出 512 通道特征图，CBAM 在此做通道+空间注意力提纯
        if config.use_cbam:
            self.cbam = CBAM(channels=512, reduction=16, kernel_size=7)
        else:
            self.cbam = None

        self.fc_features = nn.ModuleDict()
        for feature_name, num_feature in config.feature_sizes.items():
            self.fc_features[feature_name] = nn.Sequential(
                nn.Linear(self.total_features, num_feature),
                nn.SELU(inplace=True),
                nn.Linear(num_feature, num_feature),
            )

        self.fc_conf = nn.ModuleDict()
        for feature_name, num_feature in config.feature_sizes.items():
            self.fc_conf[feature_name] = nn.Sequential(
                nn.Linear(self.total_features, num_feature),
                nn.SELU(inplace=True),
                nn.Linear(num_feature, 1),
            )

    def forward(self, eye_image):
        # 手动展开 ResNet 前向传播，以便在 layer4 和 avgpool 之间插入 CBAM
        x = self.cnn_layers.conv1(eye_image)
        x = self.cnn_layers.bn1(x)
        x = self.cnn_layers.relu(x)
        x = self.cnn_layers.maxpool(x)

        x = self.cnn_layers.layer1(x)   # [B, 64,  H/4,  W/4]
        x = self.cnn_layers.layer2(x)   # [B, 128, H/8,  W/8]
        x = self.cnn_layers.layer3(x)   # [B, 256, H/16, W/16]
        x = self.cnn_layers.layer4(x)   # [B, 512, H/32, W/32]

        # CBAM 在 ResNet 全部残差层结束后、全局池化前做注意力加权
        if self.cbam is not None:
            x = self.cbam(x)             # [B, 512, H/32, W/32]

        x = self.cnn_layers.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)         # [B, 512]
        x = self.cnn_layers.fc(x)       # [B, total_features]

        out_features = OrderedDict()
        out_confs = OrderedDict()
        for feature_name in config.feature_sizes.keys():
            out_features[feature_name] = self.fc_features[feature_name](x)
            out_confs[feature_name] = torch.squeeze(self.fc_conf[feature_name](x), dim=-1)
        return out_features, out_confs


class BaselineGenerator(Generator):
    def __init__(self, input_num_feature, generator_num_feature=64):
        super(BaselineGenerator, self).__init__()

        self.input_num_feature = input_num_feature
        self.generator_num_feature = generator_num_feature

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_num_feature, generator_num_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_num_feature * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(generator_num_feature * 8, generator_num_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_num_feature * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(generator_num_feature * 4, generator_num_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_num_feature * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(generator_num_feature * 2, generator_num_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_num_feature),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(generator_num_feature , int(generator_num_feature/2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(generator_num_feature/2)),
            nn.ReLU(True),
            # state size. (ngf/2) x 64 x 64
            nn.ConvTranspose2d(int(generator_num_feature/2), 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

        # initialization
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        return self.main(input)
