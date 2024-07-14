import torch.nn as nn
import math

class AlexNetBN_FCN(nn.Module):

    def __init__(self, output_dim=256, stride=[4, 2, 2, 2], dilation=[1, 1], padding=[1, 1]):
        super(AlexNetBN_FCN, self).__init__()
        BN = nn.BatchNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=stride[0], padding=5),
            BN(96),
            nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=stride[1], padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            BN(256),
            nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=stride[2], padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            BN(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=padding[0], dilation=dilation[0]),
            BN(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=padding[1], dilation=dilation[1]),
            BN(256),
            nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=stride[3], padding=1)

        self.fc6 = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=3, stride=1, padding=1),
            BN(4096),
            nn.ReLU(inplace=True))
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
            BN(4096),
            nn.ReLU(inplace=True))
        self.drop7 = nn.Dropout(0.5)
        self.conv8 = nn.Conv2d(4096, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                scale = math.sqrt(2. / fan_in)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, ret_feat=False):
        if ret_feat:
            raise NotImplemented
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)
        x = self.conv8(x)
        return x

def alexnet_fcn_32x(output_dim, pretrained=False, **kwargs):
    assert pretrained == False
    model = AlexNetBN_FCN(output_dim=output_dim, **kwargs)
    return model

def alexnet_fcn_8x(output_dim, use_ppm=False, pretrained=False, **kwargs):
    assert pretrained == False
    model = AlexNetBN_FCN(output_dim=output_dim, stride=[2, 2, 2, 1], **kwargs)
    return model
