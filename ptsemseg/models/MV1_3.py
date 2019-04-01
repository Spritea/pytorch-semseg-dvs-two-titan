import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
from torch.nn import functional as F

__all__ = ['MV1_ResNet', 'MV1_ResNet18', 'MV1_ResNet34', 'MV1_ResNet50', 'MV1_ResNet101',
           'MV1_ResNet152']
models_urls = {
    '101_voc': 'https://cloudstor.aarnet.edu.au/plus/s/Owmttk9bdPROwc6/download',
    '18_imagenet': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34_imagenet': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50_imagenet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_same_bn(same_planes):
    return nn.Sequential(nn.Conv2d(same_planes, same_planes, kernel_size=3, stride=1, padding=1, bias=False),
                         nn.ReLU(inplace=True))


class MultiResolutionFuse(nn.Module):
    def __init__(self, in_size, out_size):
        super(MultiResolutionFuse, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False)

    def forward(self, input_low, input_high):
        high_size = input_high.size()[2:]
        # low channel usually > high channel
        input_low = self.conv(input_low)
        upsample_low = F.upsample(input_low, high_size, mode='bilinear')
        cat = torch.cat([upsample_low, input_high], dim=1)
        return cat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FPA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FPA, self).__init__()

        self.c7_1 = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.c5_1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.c3_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.c7_2 = nn.Conv2d(out_channel, out_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.c5_2 = nn.Conv2d(out_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.c3_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x7_1=self.c7_1(x)
        x7_1=self.bn(x7_1)
        x7_1=self.relu(x7_1)
        x7_2=self.c7_2(x7_1)
        x7_2=self.bn(x7_2)

        x5_1 = self.c5_1(x)
        x5_1 = self.bn(x5_1)
        x5_1 = self.relu(x5_1)
        x5_2 = self.c5_2(x5_1)
        x5_2 = self.bn(x5_2)

        x3_1 = self.c3_1(x)
        x3_1 = self.bn(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.c3_2(x3_1)
        x3_2 = self.bn(x3_2)

        out=torch.cat([x7_2,x5_2,x3_2],dim=1)
        return out
#add FPA
class MV1_3_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(MV1_3_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # only for >=res50

        self.fpa=FPA(2048,512)

        self.fuse43 = MultiResolutionFuse(512 * 3, 1024)
        self.post_proc43 = conv3x3_same_bn(1024)
        self.fuse32 = MultiResolutionFuse(1024 * 2, 512)
        self.post_proc32 = conv3x3_same_bn(512)
        self.fuse21 = MultiResolutionFuse(512 * 2, 256)
        self.post_proc21 = conv3x3_same_bn(256)

        self.class_conv = nn.Conv2d(256 * 2, num_classes, kernel_size=3, stride=1,
                                    padding=1, bias=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ori_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        x_l4_final=self.fpa(l4)

        x_fuse43 = self.fuse43(x_l4_final, l3)
        # post
        x_fuse32 = self.fuse32(x_fuse43, l2)
        x_fuse21 = self.fuse21(x_fuse32, l1)
        x = self.class_conv(x_fuse21)
        x = F.upsample(x, ori_size, mode='bilinear')

        return x


def MV1_3_ResNet18(num_classes, pretrained=False, **kwargs):
    """Constructs a MV1_ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MV1_3_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs, num_classes=num_classes)
    if pretrained:
        key = '18_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def MV1_3_ResNet34(num_classes, pretrained=False, **kwargs):
    """Constructs a MV1_ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MV1_3_ResNet(BasicBlock, [3, 4, 6, 3], **kwargs, num_classes=num_classes)
    if pretrained:
        key = '34_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def MV1_3_ResNet50(num_classes, pretrained=True, **kwargs):
    """Constructs a MV1_ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MV1_3_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, num_classes=num_classes)
    if pretrained:
        key = '50_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
        print("load imagenet res50")
    return model


def MV1_3_ResNet101(num_classes, pretrained=False, **kwargs):
    """Constructs a MV1_ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MV1_3_ResNet(Bottleneck, [3, 4, 23, 3], **kwargs, num_classes=num_classes)
    if pretrained:
        key = '101_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def MV1_3_ResNet152(num_classes, pretrained=False, **kwargs):
    """Constructs a MV1_ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MV1_3_ResNet(Bottleneck, [3, 8, 36, 3], **kwargs, num_classes=num_classes)
    if pretrained:
        key = '152_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    return model