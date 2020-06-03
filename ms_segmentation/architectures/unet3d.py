# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script that defines the basic 3D patch-based U-Net architecture
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      Script modified from code provided by Dr. Sergi Valverde during a seminar that took place in the UdG in 2019
#
# --------------------------------------------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch


class SingleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.single_conv(x)

class DoubleConv3D_alt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SingleConv3D(in_channels, out_channels)
        self.conv2 = SingleConv3D(out_channels, out_channels)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1, x2

class Down3D_alt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.double_conv = DoubleConv3D_alt(in_channels, out_channels)

    def forward(self, x):
        x1 = self.maxpool(x)
        x2, x3 = self.double_conv(x1)
        return x2, x3

class Up3D_alt(nn.Module):
    """Upscaling then double conv"""
    # Lines 63(64), 76 (77) to change to hybrid 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv1 = SingleConv3D(in_channels, in_channels//2)
        self.conv2 = SingleConv3D(in_channels, out_channels) #normal double skip connections
        #self.conv2 = SingleConv3D(in_channels//2, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1) # First skip connection (concatenation)
        x4 = self.conv1(x)
        #x5 = self.conv2(x3+x4) # Second skip connection (sum)
        x5 = self.conv2(torch.cat([x3,x4], dim=1)) # second skip connection (concatenation) - green
        return x5


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




# ------------------------------------------------------------------------------------------
# Basic UNet3D
# ------------------------------------------------------------------------------------------

class UNet_3D_double_skip_hybrid(nn.Module):
    """
    UNet with double skip connections (sum and concatenation)
    Changes: 
        Blocks implemented according to class blocks defined in unet3d file. Double convolutions used along with BN
    """

    def __init__(self, n_channels, n_classes, bilinear=True):

        super(UNet_3D_double_skip_hybrid, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D_alt(n_channels, 32)
        self.down1 = Down3D_alt(32, 64)
        self.down2 = Down3D_alt(64, 128)
        self.down3 = Down3D_alt(128, 256)
        self.down4 = Down3D_alt(256, 256)
        self.up1 = Up3D_alt(512, 128, bilinear)
        self.up2 = Up3D_alt(256, 64, bilinear)
        self.up3 = Up3D_alt(128, 32, bilinear)
        self.up4 = Up3D_alt(64, 32, bilinear)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        #x eg (10,3,32,32,32)
        x1, x2 = self.inc(x) # (10,32,32,32,32)
        x3, x4 = self.down1(x2) # (10,64,16,16,16)
        x5, x6 = self.down2(x4) # (10,128,8,8,8)
        x7, x8 = self.down3(x6) # (10,256,4,4,4)
        _, x10 = self.down4(x8) # (10,256,2,2,2)

        x = self.up1(x10, x8, x7) # (10,128,4,4,4)
        x = self.up2(x, x6, x5) # (10,64,8,8,8)
        x = self.up3(x, x4, x3) # (10,32,16,16,16)
        x = self.up4(x, x2, x1) # (10,32,32,32,32)

        logits = F.softmax(self.outc(x), dim=1) # (10,2,32,32,32)
        return logits 


class UNet_3D_alt(nn.Module):
    """
    Basic UNet
    Changes: 
        Blocks implemented according to class blocks defined in unet3d file. Double convolutions used along with BN
    """

    def __init__(self, n_channels, n_classes, bilinear=True):

        super(UNet_3D_alt, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 32)
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        self.down4 = Down3D(256, 256)
        self.up1 = Up3D(512, 128, bilinear)
        self.up2 = Up3D(256, 64, bilinear)
        self.up3 = Up3D(128, 32, bilinear)
        self.up4 = Up3D(64, 32, bilinear)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        #x eg (10,3,32,32,32)
        x1 = self.inc(x) # (10,32,32,32,32)
        x2 = self.down1(x1) # (10,64,16,16,16)
        x3 = self.down2(x2) # (10,128,8,8,8)
        x4 = self.down3(x3) # (10,256,4,4,4)
        x5 = self.down4(x4) # (10,256,2,2,2)

        x = self.up1(x5, x4) # (10,128,4,4,4)
        x = self.up2(x, x3) # (10,64,8,8,8)
        x = self.up3(x, x2) # (10,32,16,16,16)
        x = self.up4(x, x1) # (10,32,32,32,32)

        logits = F.softmax(self.outc(x), dim=1) # (10,2,32,32,32)
        return logits 


class UNet_3D_double_encoder(nn.Module):
    """
    Basic UNet with double encoder (longitudinal + cross-sectional)
    Changes: 
        Blocks implemented according to class blocks defined in unet3d file. Double convolutions used along with BN
    """

    def __init__(self, n_channels_t, n_channels_m, n_classes, bilinear=True):
        # n_channels_t -> Number of timepoints
        # n_channels_m -> Number of sequences
        super(UNet_3D_double_encoder, self).__init__()

        self.n_channels1 = n_channels_t
        self.n_channels2 = n_channels_m
        self.n_classes = n_classes
        self.bilinear = bilinear

        # first encoder
        self.inc_e1 = DoubleConv3D(n_channels_t, 16)
        self.down1_e1 = Down3D(16, 32)
        self.down2_e1 = Down3D(32, 64)
        self.down3_e1 = Down3D(64, 128)
        self.down4_e1 = Down3D(128, 128)

        # second encoder
        self.inc_e2 = DoubleConv3D(n_channels_m, 16)
        self.down1_e2 = Down3D(16, 32)
        self.down2_e2 = Down3D(32, 64)
        self.down3_e2 = Down3D(64, 128)
        self.down4_e2 = Down3D(128, 128)

        # decoder
        self.up1 = Up3D(512, 128, bilinear)
        self.up2 = Up3D(256, 64, bilinear)
        self.up3 = Up3D(128, 32, bilinear)
        self.up4 = Up3D(64, 16, bilinear)
        self.outc = OutConv3D(16, n_classes)

    def forward(self, x):
        #x eg (10,3,4,32,32,32)

        # encoder 1
        # input to encoder 1: (10,3,32,32,32) -> Only flair, all timepoints
        x1_1 = self.inc_e1(x[:,:,0,:,:,:]) # (10,16,32,32,32)
        x2_1 = self.down1_e1(x1_1) # (10,32,16,16,16)
        x3_1 = self.down2_e1(x2_1) # (10,64,8,8,8)
        x4_1 = self.down3_e1(x3_1) # (10,128,4,4,4)
        x5_1 = self.down4_e1(x4_1) # (10,128,2,2,2)

        # encoder 2
        # input to encoder 2: (10,4,32,32,32) -> 4 sequences, only timepoint in the middle
        x1_2 = self.inc_e2(x[:,1,:,:,:,:]) # (10,16,32,32,32)
        x2_2 = self.down1_e2(x1_2) # (10,32,16,16,16)
        x3_2 = self.down2_e2(x2_2) # (10,64,8,8,8)
        x4_2 = self.down3_e2(x3_2) # (10,128,4,4,4)
        x5_2 = self.down4_e2(x4_2) # (10,128,2,2,2)

        # decoder
        x = self.up1(torch.cat([x5_1, x5_2], dim = 1), torch.cat([x4_1, x4_2], dim=1)) # (10,128,4,4,4)
        x = self.up2(x, torch.cat([x3_1, x3_2], dim=1)) # (10,64,8,8,8)
        x = self.up3(x, torch.cat([x2_1, x2_2], dim=1)) # (10,32,16,16,16)
        x = self.up4(x, torch.cat([x1_1, x1_2], dim=1)) # (10,32,32,32,32)

        logits = F.softmax(self.outc(x), dim=1) # (10,2,32,32,32)
        return logits 


class UNet_3D_double_encoder_v2(nn.Module):
    """
    Basic UNet with double encoder (longitudinal + cross-sectional)
    Changes: 
        Blocks implemented according to class blocks defined in unet3d file. Double convolutions used along with BN
    """

    def __init__(self, n_channels_t, n_channels_m, n_classes, bilinear=True):
        # n_channels_t -> Number of timepoints
        # n_channels_m -> Number of sequences
        super(UNet_3D_double_encoder_v2, self).__init__()

        self.n_channels1 = n_channels_t
        self.n_channels2 = n_channels_m
        self.n_classes = n_classes
        self.bilinear = bilinear

        # first encoder
        self.inc_e1 = DoubleConv3D(n_channels_t, 32)
        self.down1_e1 = Down3D(32, 64)
        self.down2_e1 = Down3D(64, 128)
        self.down3_e1 = Down3D(128, 256)
        self.down4_e1 = Down3D(256, 256)

        # second encoder
        self.inc_e2 = DoubleConv3D(n_channels_m, 32)
        self.down1_e2 = Down3D(32, 64)
        self.down2_e2 = Down3D(64, 128)
        self.down3_e2 = Down3D(128, 256)
        self.down4_e2 = Down3D(256, 256)

        # decoder
        self.up1 = Up3D(1024, 256, bilinear)
        self.up2 = Up3D(512, 128, bilinear)
        self.up3 = Up3D(256, 64, bilinear)
        self.up4 = Up3D(128, 32, bilinear)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        #x eg (10,3,4,32,32,32)

        # encoder 1
        # input to encoder 1: (10,3,32,32,32) -> Only flair, all timepoints
        x1_1 = self.inc_e1(x[:,:,0,:,:,:]) # (10,16,32,32,32)
        x2_1 = self.down1_e1(x1_1) # (10,32,16,16,16)
        x3_1 = self.down2_e1(x2_1) # (10,64,8,8,8)
        x4_1 = self.down3_e1(x3_1) # (10,128,4,4,4)
        x5_1 = self.down4_e1(x4_1) # (10,128,2,2,2)

        # encoder 2
        # input to encoder 2: (10,4,32,32,32) -> 4 sequences, only timepoint in the middle
        x1_2 = self.inc_e2(x[:,1,:,:,:,:]) # (10,16,32,32,32)
        x2_2 = self.down1_e2(x1_2) # (10,32,16,16,16)
        x3_2 = self.down2_e2(x2_2) # (10,64,8,8,8)
        x4_2 = self.down3_e2(x3_2) # (10,128,4,4,4)
        x5_2 = self.down4_e2(x4_2) # (10,128,2,2,2)

        # decoder
        x = self.up1(torch.cat([x5_1, x5_2], dim = 1), torch.cat([x4_1, x4_2], dim=1)) # (10,128,4,4,4)
        x = self.up2(x, torch.cat([x3_1, x3_2], dim=1)) # (10,64,8,8,8)
        x = self.up3(x, torch.cat([x2_1, x2_2], dim=1)) # (10,32,16,16,16)
        x = self.up4(x, torch.cat([x1_1, x1_2], dim=1)) # (10,32,32,32,32)

        logits = F.softmax(self.outc(x), dim=1) # (10,2,32,32,32)
        return logits 
