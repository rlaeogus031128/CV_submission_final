import torch
import torch.nn as nn
import models.submission_20226275.modules as modules


class submission_20226275(nn.Module): 
    def __init__(self, in_channels, num_classes):
        super().__init__()
        U = 12

        self.mres1 = modules.MultiResBlock(in_channels, U)
        self.scse1 = modules.SCSEBlock(U)
        self.pool1 = nn.MaxPool2d(2)

        self.mres2 = modules.MultiResBlock(U, U * 2)
        self.scse2 = modules.SCSEBlock(U * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.mres3 = modules.MultiResBlock(U * 2, U * 4)
        self.scse3 = modules.SCSEBlock(U * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mres4 = modules.MultiResBlock(U * 4, U * 8)
        self.scse4 = modules.SCSEBlock(U * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.mres5 = modules.MultiResBlock(U * 8, U * 16)

        self.res1 = modules.ResPath(U, 4)
        self.res2 = modules.ResPath(U * 2, 3)
        self.res3 = modules.ResPath(U * 4, 2)
        self.res4 = modules.ResPath(U * 8, 1)

        self.dec4 = modules.DecoderBlock(U * 16, U * 8, U * 8)
        self.dec3 = modules.DecoderBlock(U * 8, U * 4, U * 4)
        self.dec2 = modules.DecoderBlock(U * 4, U * 2, U * 2)
        self.dec1 = modules.DecoderBlock(U * 2, U, U)

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
