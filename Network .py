import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import Variable
from skimage.segmentation import slic, mark_boundaries
import random

import metrics as criterion
import os


criterion_mse = torch.nn.MSELoss()

criterion_kl = torch.nn.KLDivLoss()
def JS(gf,lf):
    kl_glo = F.softmax(gf, dim=-1)
    kl_loc = F.softmax(lf, dim=-1)
    kl_glo_log = F.log_softmax(gf, dim=-1)
    kl_loc_log = F.log_softmax(lf, dim=-1)
    loss_mid_net = (criterion_kl(kl_glo_log, kl_loc) + criterion_kl(kl_loc_log, kl_glo)) / 2
    return loss_mid_net


class Unet3D(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(Unet3D, self).__init__()

        self.DownConv1_conv1 = nn.Conv3d(inchannel,64,3,padding=1)
        self.DownConv1_norm1 = nn.GroupNorm(4, 64)
        self.DownConv1_GELU1 = nn.LeakyReLU()
        self.DownConv1_conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.DownConv1_norm2 = nn.GroupNorm(4, 64)
        self.DownConv1_GELU2 = nn.LeakyReLU()

        self.Down1 = nn.MaxPool3d((2, 2, 2))

        self.DownConv2_conv1 = nn.Conv3d(64,128,3,padding=1)
        self.DownConv2_norm1 = nn.GroupNorm(4, 128)
        self.DownConv2_GELU1 = nn.LeakyReLU()
        self.DownConv2_conv2 = nn.Conv3d(128, 128,3,padding=1)
        self.DownConv2_norm2 = nn.GroupNorm(4, 128)
        self.DownConv2_GELU2 = nn.LeakyReLU()

        self.Down2 = nn.MaxPool3d((2, 2, 2))

        self.DownConv3_conv1 = nn.Conv3d(128, 256,3,padding=1)
        self.DownConv3_norm1 = nn.GroupNorm(4, 256)
        self.DownConv3_GELU1 = nn.LeakyReLU()
        self.DownConv3_conv2 = nn.Conv3d(256, 256,3,padding=1)
        self.DownConv3_norm2 = nn.GroupNorm(4, 256)
        self.DownConv3_GELU2 = nn.LeakyReLU()

        self.Down3 = nn.MaxPool3d((2, 2, 2))

        self.botten_conv1 = nn.Conv3d(256, 512,3,padding=1)
        self.botten_norm1 = nn.GroupNorm(4, 512)
        self.botten_GELU1 = nn.LeakyReLU()
        self.botten_conv2 = nn.Conv3d(512, 512,3,padding=1)
        self.botten_norm2 = nn.GroupNorm(4, 512)
        self.botten_GELU2 = nn.LeakyReLU()


        self.UpSample3_up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        self.UpSample3_conv = nn.Conv3d(512, 256,3,padding=1)

        self.up3_conv1 = nn.Conv3d(512, 256,3,padding=1)
        self.up3_norm1 = nn.GroupNorm(4, 256)
        self.up3_GELU1 = nn.LeakyReLU()
        self.up3_conv2 = nn.Conv3d(256, 256,3,padding=1)
        self.up3_norm2 = nn.GroupNorm(4, 256)
        self.up3_GELU2 = nn.LeakyReLU()

        self.UpSample2_up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        self.UpSample2_conv = nn.Conv3d(256, 128,3,padding=1)

        self.up2_conv1 = nn.Conv3d(256, 128,3,padding=1)
        self.up2_norm1 = nn.GroupNorm(4, 128)
        self.up2_GELU1 = nn.LeakyReLU()
        self.up2_conv2 = nn.Conv3d(128, 128,3,padding=1)
        self.up2_norm2 = nn.GroupNorm(4, 128)
        self.up2_GELU2 = nn.LeakyReLU()

        self.UpSample1_up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        self.UpSample1_conv = nn.Conv3d(128, 64,3,padding=1)

        self.up1_conv1 = nn.Conv3d(128, 64,3,padding=1)
        self.up1_norm1 = nn.GroupNorm(4, 64)
        self.up1_GELU1 = nn.LeakyReLU()
        self.up1_conv2 = nn.Conv3d(64, 64,3,padding=1)
        self.up1_norm2 = nn.GroupNorm(4, 64)
        self.up1_GELU2 = nn.LeakyReLU()

        self.out = nn.Conv3d(64, outchannel, 3, padding=1)

    def forward(self,x):

        b, c, d, w, h = x.shape

        ############
        dc1_1 = self.DownConv1_conv1(x)
        dc1_1 = self.DownConv1_GELU1(self.DownConv1_norm1(dc1_1))
        dc1_1 = self.DownConv1_conv2(dc1_1)
        dc1_1 = self.DownConv1_GELU2(self.DownConv1_norm2(dc1_1))
        ############
        do1 = self.Down1(dc1_1)
        ############
        dc2_1 = self.DownConv2_conv1(do1)
        dc2_1 = self.DownConv2_GELU1(self.DownConv2_norm1(dc2_1))
        dc2_1 = self.DownConv2_conv2(dc2_1)
        dc2_1 = self.DownConv2_GELU2(self.DownConv2_norm2(dc2_1))
        ############
        do2 = self.Down2(dc2_1)
        ############
        dc3_1 = self.DownConv3_conv1(do2)
        dc3_1 = self.DownConv3_GELU1(self.DownConv3_norm1(dc3_1))
        dc3_1 = self.DownConv3_conv2(dc3_1)
        dc3_1 = self.DownConv3_GELU2(self.DownConv3_norm2(dc3_1))
        ############
        do3 = self.Down3(dc3_1)
        ############
        bo_1 = self.botten_conv1(do3)
        bo_1 = self.botten_GELU1(self.botten_norm1(bo_1))
        bo_1 = self.botten_conv2(bo_1)
        bo_1 = self.botten_GELU2(self.botten_norm2(bo_1))
        ############
        up3 = self.UpSample3_up(bo_1)
        a = up3
        c = (w // 4 - a.size()[3])
        c1 = (h // 4 - a.size()[4])
        cc = (d // 4 - a.size()[2])
        bypass = F.pad(a, (0, c1, 0, c, 0, cc))
        up3 = self.UpSample3_conv(bypass)
        cat2 = torch.cat((dc3_1, up3), 1)
        ############
        uc3_1 = self.up3_conv1(cat2)
        uc3_1 = self.up3_GELU1(self.up3_norm1(uc3_1))
        uc3_1 = self.up3_conv2(uc3_1)
        uc3_1 = self.up3_GELU2(self.up3_norm2(uc3_1))
        ############
        up2 = self.UpSample2_up(uc3_1)
        a = up2
        c = (w // 2 - a.size()[3])
        c1 = (h // 2 - a.size()[4])
        cc = (d//2 - a.size()[2])
        bypass = F.pad(a, (0, c1, 0, c, 0, cc))
        up2 = self.UpSample2_conv(bypass)
        cat3 = torch.cat((dc2_1, up2), 1)
        ############
        uc2_1 = self.up2_conv1(cat3)
        uc2_1 = self.up2_GELU1(self.up2_norm1(uc2_1))
        uc2_1 = self.up2_conv2(uc2_1)
        uc2_1 = self.up2_GELU2(self.up2_norm2(uc2_1))
        ############
        up1 = self.UpSample1_up(uc2_1)
        a = up1
        c = (w - a.size()[3])
        c1 = (h - a.size()[4])
        cc = (d - a.size()[2])
        bypass = F.pad(a, (0, c1, 0, c, 0, cc))
        up1 = self.UpSample1_conv(bypass)
        cat4 = torch.cat((dc1_1, up1), 1)
        ############
        uc1_1 = self.up1_conv1(cat4)
        uc1_1 = self.up1_GELU1(self.up1_norm1(uc1_1))
        uc1_1 = self.up1_conv2(uc1_1)
        uc1_1 = self.up1_GELU2(self.up1_norm2(uc1_1))
        ############
        out = self.out(uc1_1)
        out = F.softmax(out, dim=1)
        return out




if __name__ == '__main__':
    inputs = torch.randn(1, 1, 24, 24)
    model = Unet3D(1, 2)
    print(model)
    outputs, lf, gf = model(inputs,inputs)
    print("输入维度:", inputs.shape)
    print("输出维度:", outputs.shape)





