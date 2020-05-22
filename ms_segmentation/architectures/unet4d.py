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

from __future__ import division
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv4d:
    """Class for 4D convolution. From https://github.com/timothygebhard/pytorch-conv4d/blob/master/conv4d.py
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int, int],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):

        super(Conv4d, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert stride == 1, \
            'Strides other than 1 not yet implemented!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(l_k):

            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=(d_k, h_k, w_k),
                                           padding=self.padding)

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv3d_layer.weight)
            if self.bias_initializer is not None:
                self.bias_initializer(conv3d_layer.bias)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        (l_o, d_o, h_o, w_o) = (l_i + 2 * self.padding - l_k + 1,
                                d_i + 2 * self.padding - d_k + 1,
                                h_i + 2 * self.padding - h_k + 1,
                                w_i + 2 * self.padding - w_k + 1)

        # Output tensors for each 3D frame
        frame_results = l_o * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):

            for j in range(l_i):

                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue

                frame_conv3d = \
                    self.conv3d_layers[i](input[:, :, j, :]
                                          .view(b, c_i, d_i, h_i, w_i))

                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d

        return torch.stack(frame_results, dim=2)


class MaxPool4d:

    def __init__(self):
        self.mp = nn.MaxPool3d(2)

    def forward(self, input):
        #input should have size (B, T, M, H, W, D)
        output = torch.zeros(input.size(0), input.size(1), input.size(2), input.size(3)//2, input.size(4)//2, input.size(5)//2)
        for i in range(input.size(1)):
            output[:,i] = self.mp(input[:,i])
        new_output = torch.zeros(input.size(0), input.size(1), input.size(2)//2, input.size(3)//2, input.size(4)//2, input.size(5)//2)
        cnt = 0
        for i in range(0, output.size(2), 2):
            new_output[:,:,cnt,:] = torch.max(output[:,:,i:i+1,:], dim= 2)[0]
            cnt += 1
        return new_output


class Down4D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            MaxPool4d(),
            DoubleConv4D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DoubleConv4D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv4d(in_channels, out_channels, kernel_size=(3,3,3,3), padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            Conv4d(out_channels, out_channels, kernel_size=(3,3,3,3), padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

if __name__ == "__main__":

    arr = torch.randn(10,3,4,32,32,32)
    mp = DoubleConv4D(3, 6)
    hola = mp.forward(arr)
    layer = Conv4d(in_channels = 3, out_channels = 16, kernel_size = (3,3,3,3))
    u = layer.forward(arr)