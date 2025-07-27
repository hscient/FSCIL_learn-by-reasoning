import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock12(nn.Module):
    """BasicBlock for ResNet12 - with 3 conv layers and dropout support"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
        super(BasicBlock12, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class BasicBlock18(nn.Module):
    """BasicBlock for ResNet18 - standard 2 conv layers"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock18, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet12(nn.Module):
    def __init__(self, widths=(64, 160, 320, 640), drop_rate=0.0, drop_rate_interm=0.0):
        """
        MetaOptNet style ResNet12

        Args:
            widths: Channel widths for each stage. Default is MetaOptNet style [64, 160, 320, 640]
                   Can also use TADAM style [64, 128, 256, 512]
            drop_rate: Dropout rate for final layer
            drop_rate_interm: Dropout rate for intermediate blocks
        """
        super(ResNet12, self).__init__()

        # MetaOptNet uses input channels of 3 for RGB images
        self.in_planes = 3
        block = BasicBlock12

        # Build 4 stages, each with stride=2 (total 16x downsampling)
        self.stage1 = self._make_layer(block, widths[0], stride=2, drop_rate=drop_rate_interm)
        self.stage2 = self._make_layer(block, widths[1], stride=2, drop_rate=drop_rate_interm)
        self.stage3 = self._make_layer(block, widths[2], stride=2, drop_rate=drop_rate_interm)
        self.stage4 = self._make_layer(block, widths[3], stride=2, drop_rate=drop_rate_interm)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

        # Output dimension
        self.out_dim = widths[3]

        # Initialize weights
        self._init_weights()

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, drop_rate))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, normalize=True):
        # Forward through 4 stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global average pooling
        x = self.avgpool(x)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)

        # Apply dropout
        x = self.dropout(x)

        # L2 normalize for cosine classifier (maintaining compatibility)
        return F.normalize(x, dim=1) if normalize else x

    def forward_conv(self, x):
        """Forward pass without final pooling/normalization (for feature extraction)"""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


class ResNet18(nn.Module):
    """
    Modified ResNet18 for CIFAR-100 in FSCIL research.
    Changes from standard torchvision ResNet18:
    - conv1: kernel_size=7x7 stride=2 -> 3x3 stride=1
    - maxpool: removed
    - FC layer removed, returns (B, 512) normalized features.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock18, 64, 2)
        self.layer2 = self._make_layer(BasicBlock18, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock18, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock18, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512 * BasicBlock18.expansion

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, normalize=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)
        return F.normalize(x, dim=1) if normalize else x


# Example usage:
if __name__ == "__main__":
    # Test ResNet12 with MetaOptNet style channels
    model12 = ResNet12(widths=(64, 160, 320, 640))
    print(f"ResNet12 output dim: {model12.out_dim}")

    # Test ResNet18
    model18 = ResNet18()
    print(f"ResNet18 output dim: {model18.out_dim}")

    # Test forward pass for both
    x = torch.randn(4, 3, 84, 84)  # miniImageNet size
    out12 = model12(x)
    out18 = model18(x)
    print(f"ResNet12 output shape: {out12.shape}")  # Should be (4, 640) normalized
    print(f"ResNet18 output shape: {out18.shape}")  # Should be (4, 512) normalized