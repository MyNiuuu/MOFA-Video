import torch.nn as nn
import math

class ShallowNet(nn.Module):

    def __init__(self, input_dim=4, output_dim=16, stride=[2, 2, 2]):
        super(ShallowNet, self).__init__()
        global BN

        BN = nn.BatchNorm2d

        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=5, stride=stride[0], padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=stride[1], stride=stride[1]),
            nn.Conv2d(16, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=stride[2], stride=stride[2]),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                scale = math.sqrt(2. / fan_in)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                if not m.bias is None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        return x


def shallownet8x(output_dim):
    model = ShallowNet(output_dim=output_dim, stride=[2,2,2])
    return model

def shallownet32x(output_dim, **kwargs):
    model = ShallowNet(output_dim=output_dim, stride=[2,2,8])
    return model



