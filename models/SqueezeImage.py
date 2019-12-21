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


class FireRes(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, bn_d=0.1):
        super(FireRes, self).__init__()

        self.Fire = Fire(inplanes, squeeze_planes,
                         expand1x1_planes, expand3x3_planes, bn_d=0.1)
        self.skip = nn.Conv2d(in_channels=inplanes, out_channels=expand3x3_planes + expand1x1_planes, kernel_size=1)

    def forward(self, inp):
        x = self.Fire(inp)
        skip = self.skip(inp)
        return x + skip


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


class Decoder(nn.Module):
    """

    copied from: https://github.com/PRBonn/lidar-bonnetal
    """

    def __init__(self, OS, bn_d, encoder_feature_depth, drop_prob):
        super(Decoder, self).__init__()
        self.OS = OS

        # decoder
        # decoder
        self.firedec1 = FireUp(encoder_feature_depth, 64, 128, 128, bn_d=bn_d)
        self.firedec2 = FireUp(256, 32, 64, 64, bn_d=bn_d)
        self.firedec3 = FireUp(128, 32, 64, 64, bn_d=bn_d)
        self.firedec4 = FireUp(128, 16, 64, 64, bn_d=bn_d)
        self.firedec5 = FireUp(128, 16, 32, 32, bn_d=bn_d)

        self.dropout = nn.Dropout2d(drop_prob)

        # last channels
        self.last_channels = 64

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os]  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = self.OS

        # run layers
        x, skips, os = self.run_layer(x, self.firedec1, skips, os)
        x, skips, os = self.run_layer(x, self.firedec2, skips, os)
        x, skips, os = self.run_layer(x, self.firedec3, skips, os)
        x, skips, os = self.run_layer(x, self.firedec4, skips, os)
        x, skips, os = self.run_layer(x, self.firedec5, skips, os)

        x = self.dropout(x)

        return x

    def get_last_depth(self):
        return self.last_channels


class Encoder(nn.Module):
    """
    copied from: https://github.com/PRBonn/lidar-bonnetal
    """

    def __init__(self, bn_d, drop_prob):
        # Call the super constructor
        super(Encoder, self).__init__()
        self.bn_d = bn_d
        self.drop_prob = drop_prob

        # last channels
        self.last_channels = 512

        # encoder
        self.fire1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
                                   FireRes(64, 16, 64, 64, bn_d=self.bn_d),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d))

        self.fire2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d))
        self.fire3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d))
        self.fire4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                                   FireRes(128, 32, 128, 128, bn_d=self.bn_d),
                                   FireRes(256, 32, 128, 128, bn_d=self.bn_d))
        self.fire5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
                                   FireRes(256, 48, 192, 192, bn_d=self.bn_d),
                                   FireRes(384, 48, 192, 192, bn_d=self.bn_d),
                                   FireRes(384, 64, 256, 256, bn_d=self.bn_d),
                                   FireRes(512, 64, 256, 256, bn_d=self.bn_d))

        # output
        self.dropout = nn.Dropout2d(self.drop_prob)



    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # store for skip connections
        skips = {}
        os = 1

        x, skips, os = self.run_layer(x, self.fire1, skips, os)
        x = self.dropout(x)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire2, skips, os)
        x = self.dropout(x)

        x, skips, os = self.run_layer(x, self.fire3, skips, os)
        x = self.dropout(x)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire4, skips, os)
        x = self.dropout(x)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire5, skips, os)
        x = self.dropout(x)
        return x, skips


class SqueezeImage(nn.Module):

    def __init__(self, num_classes):
        super(SqueezeImage, self).__init__()
        self.num_classes = num_classes

        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder = Encoder(bn_d=0.1, drop_prob=0.3)

        self.decoder = Decoder(encoder_feature_depth=self.encoder.last_channels, OS=32, bn_d=0.1, drop_prob=0.3)

        self.exit_flow = nn.Sequential(
            nn.Conv2d(64, self.num_classes, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, image):
        x = self.entry_flow(image)
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.exit_flow(x)
        return x

