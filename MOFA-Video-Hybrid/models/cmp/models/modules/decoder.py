import torch
import torch.nn as nn
import math

class MotionDecoderPlain(nn.Module):

    def __init__(self, input_dim=512, output_dim=2, combo=[1,2,4]):
        super(MotionDecoderPlain, self).__init__()
        BN = nn.BatchNorm2d

        self.combo = combo
        for c in combo:
            assert c in [1,2,4,8], "invalid combo: {}".format(combo)

        if 1 in combo:
            self.decoder1 = nn.Sequential(
                nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
                BN(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                BN(128),
                nn.ReLU(inplace=True))

        if 2 in combo:
            self.decoder2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
                BN(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                BN(128),
                nn.ReLU(inplace=True))

        if 4 in combo:
            self.decoder4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
                BN(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                BN(128),
                nn.ReLU(inplace=True))

        if 8 in combo:
            self.decoder8 = nn.Sequential(
                nn.MaxPool2d(kernel_size=8, stride=8),
                nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
                BN(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                BN(128),
                nn.ReLU(inplace=True))

        self.head = nn.Conv2d(128 * len(self.combo), output_dim, kernel_size=1, padding=0)

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
        
        cat_list = []
        if 1 in self.combo:
            x1 = self.decoder1(x)
            cat_list.append(x1)
        if 2 in self.combo:
            x2 = nn.functional.interpolate(
                self.decoder2(x), size=(x.size(2), x.size(3)),
                mode="bilinear", align_corners=True)
            cat_list.append(x2)
        if 4 in self.combo:
            x4 = nn.functional.interpolate(
                self.decoder4(x), size=(x.size(2), x.size(3)),
                mode="bilinear", align_corners=True)
            cat_list.append(x4)
        if 8 in self.combo:
            x8 = nn.functional.interpolate(
                self.decoder8(x), size=(x.size(2), x.size(3)),
                mode="bilinear", align_corners=True)
            cat_list.append(x8)
           
        cat = torch.cat(cat_list, dim=1)
        flow = self.head(cat)
        return flow


class MotionDecoderSkipLayer(nn.Module):

    def __init__(self, input_dim=512, output_dim=2, combo=[1,2,4,8]):
        super(MotionDecoderSkipLayer, self).__init__()

        BN = nn.BatchNorm2d

        self.decoder1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.decoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.decoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.decoder8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.fusion8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            BN(256),
            nn.ReLU(inplace=True))

        self.skipconv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))
        self.fusion4 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.skipconv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            BN(32),
            nn.ReLU(inplace=True))
        self.fusion2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, kernel_size=3, padding=1),
            BN(64),
            nn.ReLU(inplace=True))

        self.head = nn.Conv2d(64, output_dim, kernel_size=1, padding=0)

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

    def forward(self, x, skip_feat):
        layer1, layer2, layer4 = skip_feat

        x1 = self.decoder1(x)
        x2 = nn.functional.interpolate(
            self.decoder2(x), size=(x1.size(2), x1.size(3)),
            mode="bilinear", align_corners=True)
        x4 = nn.functional.interpolate(
            self.decoder4(x), size=(x1.size(2), x1.size(3)),
            mode="bilinear", align_corners=True)
        x8 = nn.functional.interpolate(
            self.decoder8(x), size=(x1.size(2), x1.size(3)),
            mode="bilinear", align_corners=True)
        cat = torch.cat([x1, x2, x4, x8], dim=1)
        f8 = self.fusion8(cat)

        f8_up = nn.functional.interpolate(
            f8, size=(layer4.size(2), layer4.size(3)),
            mode="bilinear", align_corners=True)
        f4 = self.fusion4(torch.cat([f8_up, self.skipconv4(layer4)], dim=1))

        f4_up = nn.functional.interpolate(
            f4, size=(layer2.size(2), layer2.size(3)),
            mode="bilinear", align_corners=True)
        f2 = self.fusion2(torch.cat([f4_up, self.skipconv2(layer2)], dim=1))

        flow = self.head(f2)
        return flow


class MotionDecoderFlowNet(nn.Module):

    def __init__(self, input_dim=512, output_dim=2, combo=[1,2,4,8]):
        super(MotionDecoderFlowNet, self).__init__()
        global BN

        BN = nn.BatchNorm2d

        self.decoder1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.decoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.decoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.decoder8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, stride=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(inplace=True))

        self.fusion8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            BN(256),
            nn.ReLU(inplace=True))

        # flownet head
        self.predict_flow8 = predict_flow(256, output_dim)
        self.predict_flow4 = predict_flow(384 + output_dim, output_dim)
        self.predict_flow2 = predict_flow(192 + output_dim, output_dim)
        self.predict_flow1 = predict_flow(67 + output_dim, output_dim)

        self.upsampled_flow8_to_4 = nn.ConvTranspose2d(
            output_dim, output_dim, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_2 = nn.ConvTranspose2d(
            output_dim, output_dim, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(
            output_dim, output_dim, 4, 2, 1, bias=False)

        self.deconv8 = deconv(256, 128)
        self.deconv4 = deconv(384 + output_dim, 128)
        self.deconv2 = deconv(192 + output_dim, 64)

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

    def forward(self, x, skip_feat):
        layer1, layer2, layer4 = skip_feat # 3, 64, 256

        # propagation nets
        x1 = self.decoder1(x)
        x2 = nn.functional.interpolate(
            self.decoder2(x), size=(x1.size(2), x1.size(3)),
            mode="bilinear", align_corners=True)
        x4 = nn.functional.interpolate(
            self.decoder4(x), size=(x1.size(2), x1.size(3)),
            mode="bilinear", align_corners=True)
        x8 = nn.functional.interpolate(
            self.decoder8(x), size=(x1.size(2), x1.size(3)),
            mode="bilinear", align_corners=True)
        cat = torch.cat([x1, x2, x4, x8], dim=1)
        feat8 = self.fusion8(cat) # 256

        # flownet head
        flow8 = self.predict_flow8(feat8)
        flow8_up = self.upsampled_flow8_to_4(flow8)
        out_deconv8 = self.deconv8(feat8) # 128

        concat4 = torch.cat((layer4, out_deconv8, flow8_up), dim=1) # 394 + out
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_2(flow4)
        out_deconv4 = self.deconv4(concat4) # 128

        concat2 = torch.cat((layer2, out_deconv4, flow4_up), dim=1) # 192 + out
        flow2 = self.predict_flow2(concat2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv2 = self.deconv2(concat2) # 64

        concat1 = torch.cat((layer1, out_deconv2, flow2_up), dim=1) # 67 + out
        flow1 = self.predict_flow1(concat1)
        
        return [flow1, flow2, flow4, flow8]


def predict_flow(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


