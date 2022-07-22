import torch.nn as nn
import torch
##triple attention
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)##对数据进行归一化处理，使得数据在进行Relu之前不会因为数据过大导致网络性能不稳定
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample       #下采样   conv(k=1,stride=stride)+batchnorm2d
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
        #self.attention=TripletAttention()

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        ##out = self.attention(out)##注意力
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        ##out = self.attention(out)##注意力
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = [block(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)           #*，解释器将其参数自动解包然后传递给多个变量参数

    def forward(self, x):
        x = self.conv1(x)    ##1,64,3,
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc5(x)
        x = self.bn5(x)

        return x


def resnet34(use_se=True):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=use_se)
    return model



