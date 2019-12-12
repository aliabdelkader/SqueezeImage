import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Fire(nn.Module):
    """
    copied from: https://github.com/PRBonn/lidar-bonnetal

    """

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, bn_d=0.1):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d
        self.activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

    def forward(self, x):
        x = self.activation(self.squeeze_bn(self.squeeze(x)))
        return torch.cat([
            self.activation(self.expand1x1_bn(self.expand1x1(x))),
            self.activation(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)


class FireUp(nn.Module):
    """
    copied from: https://github.com/PRBonn/lidar-bonnetal
    """

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, bn_d):
        super(FireUp, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d
        self.activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
        self.upconv = nn.ConvTranspose2d(in_channels=squeeze_planes, out_channels=squeeze_planes, kernel_size=2,
                                         stride=2, padding=0)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

    def forward(self, x):
        x = self.activation(self.squeeze_bn(self.squeeze(x)))

        x = self.activation(self.upconv(x))
        return torch.cat([
            self.activation(self.expand1x1_bn(self.expand1x1(x))),
            self.activation(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)



class SqueezeImage(nn.Module):

    def __int__(self, num_classes=20):
        super(SqueezeImage, self).__init__()
        self.num_classes = num_classes

        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.Sequential(
            Fire(inplanes=64, squeeze_planes=16, expand1x1_planes=128, expand3x3_planes=128),
            Fire(inplanes=128, squeeze_planes=32, expand1x1_planes=256, expand3x3_planes=256),
            Fire(inplanes=256, squeeze_planes=64, expand1x1_planes=512, expand3x3_planes=512),
            Fire(inplanes=512, squeeze_planes=128, expand1x1_planes=1024, expand3x3_planes=1024),

        )

        self.decoder = nn.Sequential(
            FireUp(inplanes=1024, squeeze_planes=128, expand1x1_planes=512, expand3x3_planes=512),
            FireUp(inplanes=512, squeeze_planes=64, expand1x1_planes=256, expand3x3_planes=256),
            FireUp(inplanes=256, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128),
            FireUp(inplanes=128, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64),
        )

        self.exit_flow = nn.Sequential(
            nn.Conv2d(64, self.num_classes, kernel_size=1),
            nn.Sogi
        )

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def save(self, saving_dir="results", best=False):

        saving_path = os.path.join(saving_dir, "{}.pth")
        if best:
            torch.save(self.state_dict(), saving_path.format("best"))
        else:
            torch.save(self.state_dict(), saving_path.format("model"))
