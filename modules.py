import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=12):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cse = self.cSE(x)
        sse = self.sSE(x)
        return x * cse + x * sse


class MultiResBlock(nn.Module):
    def __init__(self, in_channels, U):
        super().__init__()
        W = U

        self.conv1 = nn.Conv2d(in_channels, W // 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(W // 6, W // 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(W // 3, W // 2, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, W, kernel_size=1)

    def forward(self, x):
        a_out1 = self.conv1(x)
        a_out2 = self.conv2(a_out1)
        a_out3 = self.conv3(a_out2)
        a_out = torch.cat([a_out1, a_out2, a_out3], dim=1)
        a_spatial = self.conv1x1(x)
        a_out = a_spatial + a_out
        return a_out


class ResPath(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(F_g=in_channels // 2, F_l=skip_channels, F_int=skip_channels // 2)
        self.mres = MultiResBlock(in_channels // 2 + skip_channels, out_channels)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.mres(x)
        x = self.scse(x)
        return x


class MultiResUNetAG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        U = 12

        self.mres1 = MultiResBlock(in_channels, U)
        self.scse1 = SCSEBlock(U)
        self.pool1 = nn.MaxPool2d(2)

        self.mres2 = MultiResBlock(U, U * 2)
        self.scse2 = SCSEBlock(U * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.mres3 = MultiResBlock(U * 2, U * 4)
        self.scse3 = SCSEBlock(U * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mres4 = MultiResBlock(U * 4, U * 8)
        self.scse4 = SCSEBlock(U * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.mres5 = MultiResBlock(U * 8, U * 16)

        self.res1 = ResPath(U, 4)
        self.res2 = ResPath(U * 2, 3)
        self.res3 = ResPath(U * 4, 2)
        self.res4 = ResPath(U * 8, 1)

        self.dec4 = DecoderBlock(U * 16, U * 8, U * 8)
        self.dec3 = DecoderBlock(U * 8, U * 4, U * 4)
        self.dec2 = DecoderBlock(U * 4, U * 2, U * 2)
        self.dec1 = DecoderBlock(U * 2, U, U)

        self.final = nn.Conv2d(U, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.scse1(self.mres1(x)); p1 = self.pool1(e1)
        e2 = self.scse2(self.mres2(p1)); p2 = self.pool2(e2)
        e3 = self.scse3(self.mres3(p2)); p3 = self.pool3(e3)
        e4 = self.scse4(self.mres4(p3)); p4 = self.pool4(e4)

        b = self.mres5(p4)

        e1 = self.res1(e1)
        e2 = self.res2(e2)
        e3 = self.res3(e3)
        e4 = self.res4(e4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final(d1)
