import torch.nn as nn
import torch.nn.functional as F
import torch
from .c_lstm import ConvLSTM
from .c_gru import ConvGRU, ConvGRU3D
from .c_lstm import ConvLSTM3D
from .unet2d import DoubleConv, Down, Up, OutConv, SUp
from .unet3d import DoubleConv3D, Down3D, Up3D, OutConv3D


class UNet_ConvGRU_2D(nn.Module):
    """
    Basic U-net+ConvGRU model. UNet layers built with basic blocks without batch normalization.
    Skip connections are implemented by additions
    """

    def __init__(self, input_size, output_size):

        super(UNet_ConvGRU_2D, self).__init__()

        # conv1 down
        self.conv1 = nn.Conv2d(in_channels=input_size,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        # max-pool 1
        self.pool1 = nn.MaxPool2d((2,2))
        # conv2 down
        self.conv4 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        # max-pool 2
        self.pool2 = nn.MaxPool2d((2,2))
        # conv3 down
        self.conv7 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.conv8 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        # max-pool 3
        self.pool3 = nn.MaxPool2d((2,2))
        # conv4 down (latent space)

        #Here ConvGRU acts in latent space but with all time steps
        self.convGRU1 = ConvGRU(input_size=(20,25),
                                input_dim=128,
                                hidden_dim=[256],
                                kernel_size=(3,3),
                                num_layers=1,
                                dtype=torch.cuda.FloatTensor,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)

        # up-sample conv4
        self.up1 = nn.ConvTranspose2d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=2,
                                      stride=2)        
        # conv 1 (add up1 + conv8)
        self.conv11 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.conv12 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        # up-sample conv5
        self.up2 = nn.ConvTranspose2d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=2,
                                      stride=2)
        # conv6 (add up2 + conv5) 
        self.conv14 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv15 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        # up 3
        self.up3 = nn.ConvTranspose2d(in_channels=64,
                                      out_channels=32,
                                      kernel_size=2,
                                      stride=2)
        # conv7 (add up3 + conv2)
        self.conv17 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        # conv8 (classification)
        self.conv18 = nn.Conv3d(in_channels=3,
                               out_channels=1,
                               kernel_size=1)

        self.conv19 = nn.Conv2d(in_channels=32,
                               out_channels=output_size,
                               kernel_size=1)


    def wrapper_conv(self, the_input, layer):
        num_time_steps = the_input.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, layer.out_channels, the_input.size(-2), the_input.size(-1))).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = F.relu(layer(the_input[:,i_tp,:,:,:]))
        return the_output

    def wrapper_upconv(self, the_input, layer):
        num_time_steps = the_input.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, layer.out_channels, int(2*the_input.size(-2)), int(2*the_input.size(-1)))).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input[:,i_tp,:,:,:])
        return the_output

    def wrapper_pool(self, the_input, layer):
        num_time_steps = the_input.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, the_input.size(2), the_input.size(-2) //2, the_input.size(-1)//2)).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input[:,i_tp,:,:,:])
        return the_output


    def forward(self, x):

        #Encoder
        x1 = self.wrapper_conv(x.type('torch.cuda.FloatTensor'), self.conv1) # [B, T, 32, W, W]
        x2 = self.wrapper_conv(x1, self.conv2)
        x2p = self.wrapper_pool(x2, self.pool1)
        x4 = self.wrapper_conv(x2p, self.conv4)
        x5 = self.wrapper_conv(x4, self.conv5)
        x5p = self.wrapper_pool(x5, self.pool2)
        x7 = self.wrapper_conv(x5p, self.conv7)
        x8 = self.wrapper_conv(x7, self.conv8)
        x8p = self.wrapper_pool(x8, self.pool3)

        conv_gru = self.convGRU1(x8p)[0][0]
        #conv_gru = self.wrapper_conv(x8p, self.test)
        # decoder
        up1 = self.wrapper_upconv(conv_gru, self.up1)
        x11 = self.wrapper_conv(up1+x8, self.conv11)
        x12 = self.wrapper_conv(x11, self.conv12)
        up2 = self.wrapper_upconv(x12, self.up2)
        x14 = self.wrapper_conv(up2+x5, self.conv14)
        x15 = self.wrapper_conv(x14, self.conv15)
        up3 = self.wrapper_upconv(x15, self.up3)
        x17 = self.wrapper_conv(up3+x2, self.conv17) #[B, T, 32, H, W]
        
        # output layer (2 classes)
        # we use a softmax layer to return probabilities for each class
        x18 = F.relu(self.conv18(x17))
        x18 = x18.squeeze()
        out = F.softmax(self.conv19(x18), dim=1) 
        return out #[B, 2, H, W]





class UNet_ConvGRU_2D_alt(nn.Module):
    """
    Basic U-net model. Layers implemented with blocks from unet2d file which include double convolutions and concatenations for skip connections
    """

    def __init__(self, n_channels, n_classes, bilinear=True):

        super(UNet_ConvGRU_2D_alt, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.convGRU1 = ConvGRU(input_size=(20,25),
                                input_dim=512,
                                hidden_dim=[256],
                                kernel_size=(3,3),
                                num_layers=1,
                                dtype=torch.cuda.FloatTensor,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)



        # Define wrappers
    def wrapper_conv(self, the_input, layer, out_channels, layer_type= "Down"):
        num_time_steps = the_input.size(1)
        if layer_type == "DoubleConv":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-2), the_input.size(-1))).cuda()
        elif layer_type == "Down":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()
        else: #layer_type == OutConv
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), out_channels, the_input.size(-2), the_input.size(-1))).cuda()
            the_output = layer(the_input)
            return the_output
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input[:,i_tp,:,:,:])
        return the_output

    def wrapper_up(self, the_input1, the_input2, layer, out_channels):

        num_time_steps = the_input1.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input1.size(0), num_time_steps, out_channels, int(2*the_input1.size(-2)), int(2*the_input1.size(-1)))).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input1[:,i_tp,:,:,:], the_input2[:,i_tp,:,:,:])
        return the_output

    def forward(self, x):
        x1 = self.wrapper_conv(x, self.inc, 64, layer_type = "DoubleConv")
        x2 = self.wrapper_conv(x1, self.down1, 128)
        x3 = self.wrapper_conv(x2, self.down2, 256)
        x4 = self.wrapper_conv(x3, self.down3, 512)
        #x5 = self.wrapper_conv(x4, self.down4, 512)
        x5 = self.convGRU1(x4)[0][0]
        #

        x = self.wrapper_up(x5, x3, self.up1,128)
        x = self.wrapper_up(x, x2, self.up2, 64)
        x = self.wrapper_up(x, x1, self.up3, 64)
        #x = self.wrapper_up(x, x1, self.up4, 64)
        x = x[:,-1,:,:,:]
        logits = F.softmax(self.wrapper_conv(x, self.outc, self.n_classes, layer_type="OutConv"), dim=1)
        return logits



class UNet_ConvLSTM_2D_alt(nn.Module):
    """
    Basic U-net model + LSTM
    
    """

    def __init__(self, n_channels, n_classes, bilinear=True):

        super(UNet_ConvLSTM_2D_alt, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.convLSTM1 = ConvLSTM(input_size=(20,25),
                                input_dim=512,
                                hidden_dim=[256],
                                kernel_size=(3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.convLSTM2 = ConvLSTM(  input_size = (160,200),
                                    input_dim=64,
                                    hidden_dim=[64],
                                    kernel_size=(3,3),
                                    num_layers=1,
                                    batch_first=True,
                                    bias = True,
                                    return_all_layers=False)
        self.outc = OutConv(64, n_classes)



        # Define wrappers
    def wrapper_conv(self, the_input, layer, out_channels, layer_type= "Down"):
        num_time_steps = the_input.size(1)
        if layer_type == "DoubleConv":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-2), the_input.size(-1))).cuda()
        elif layer_type == "Down":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()
        else: #layer_type == OutConv
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), out_channels, the_input.size(-2), the_input.size(-1))).cuda()
            the_output = layer(the_input)
            return the_output
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input[:,i_tp,:,:,:])
        return the_output

    def wrapper_up(self, the_input1, the_input2, layer, out_channels):

        num_time_steps = the_input1.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input1.size(0), num_time_steps, out_channels, int(2*the_input1.size(-2)), int(2*the_input1.size(-1)))).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input1[:,i_tp,:,:,:], the_input2[:,i_tp,:,:,:])
        return the_output

    def forward(self, x):
        x1 = self.wrapper_conv(x, self.inc, 64, layer_type = "DoubleConv")
        x2 = self.wrapper_conv(x1, self.down1, 128)
        x3 = self.wrapper_conv(x2, self.down2, 256)
        x4 = self.wrapper_conv(x3, self.down3, 512)
        #x5 = self.wrapper_conv(x4, self.down4, 512)
        x5 = self.convLSTM1(x4)[0]

        x = self.wrapper_up(x5, x3, self.up1,128)
        x = self.wrapper_up(x, x2, self.up2, 64)
        x = self.wrapper_up(x, x1, self.up3, 64)
        #x = self.wrapper_up(x, x1, self.up4, 64)
        x = self.convLSTM2(x)[0]
        x = x[:,-1,:,:,:]
        logits = F.softmax(self.wrapper_conv(x, self.outc, self.n_classes, layer_type="OutConv"), dim=1)
        return logits



class UNet_ConvLSTM_Goku(nn.Module):
    """
    Basic U-net model + LSTM
    
    """

    def __init__(self, n_channels, n_classes, bilinear=True):

        super(UNet_ConvLSTM_Goku, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.convLSTM1 = ConvLSTM(input_size=(160,200),
                                input_dim=32,
                                hidden_dim=[128],
                                kernel_size=(3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.convLSTM2 = ConvLSTM(input_size=(80,100),
                                input_dim=64,
                                hidden_dim=[256],
                                kernel_size=(3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.convLSTM3 = ConvLSTM(input_size=(40,50),
                                input_dim=128,
                                hidden_dim=[512],
                                kernel_size=(3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.convLSTM4 = ConvLSTM(input_size=(20,25),
                                input_dim=256,
                                hidden_dim=[512],
                                kernel_size=(3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        
        self.convLSTM5 = ConvLSTM(  input_size = (160,200),
                                    input_dim=64,
                                    hidden_dim=[64],
                                    kernel_size=(3,3),
                                    num_layers=1,
                                    batch_first=True,
                                    bias = True,
                                    return_all_layers=False)
        self.outc = OutConv(64, n_classes)



        # Define wrappers
    def wrapper_conv(self, the_input, layer, out_channels, layer_type= "Down"):
        num_time_steps = the_input.size(1)
        if layer_type == "DoubleConv":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-2), the_input.size(-1))).cuda()
        elif layer_type == "Down":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()
        else: #layer_type == OutConv
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), out_channels, the_input.size(-2), the_input.size(-1))).cuda()
            the_output = layer(the_input)
            return the_output
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input[:,i_tp,:,:,:])
        return the_output

    def wrapper_up(self, the_input1, the_input2, layer, out_channels):

        num_time_steps = the_input1.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input1.size(0), num_time_steps, out_channels, int(2*the_input1.size(-2)), int(2*the_input1.size(-1)))).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:] = layer(the_input1[:,i_tp,:,:,:], the_input2[:,i_tp,:,:,:], sum=True)
        return the_output

    def forward(self, x):
        x1 = self.wrapper_conv(x, self.inc, 32, layer_type = "DoubleConv")
        x2 = self.wrapper_conv(x1, self.down1, 64)
        x3 = self.wrapper_conv(x2, self.down2, 128)
        x4 = self.wrapper_conv(x3, self.down3, 256)
        x5 = self.convLSTM4(x4)[0]
        x = self.wrapper_up(x5, self.convLSTM3(x3)[0], self.up1,256)
        x = self.wrapper_up(x, self.convLSTM2(x2)[0], self.up2, 128)
        x = self.wrapper_up(x, self.convLSTM1(x1)[0], self.up3, 64)
        x = self.convLSTM5(x)[0]
        x = x[:,-1,:,:,:] #Take last time point
        logits = F.softmax(self.wrapper_conv(x, self.outc, self.n_classes, layer_type="OutConv"), dim=1)
        return logits





#3D



class UNet_ConvGRU_3D_1(nn.Module):
    """
    Basic U-net+ConvGRU model for 3D
    """

    def __init__(self, input_size, output_size):

        super(UNet_ConvGRU_3D_1, self).__init__()

        # conv1 down
        self.conv1 = nn.Conv3d(in_channels=input_size,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        # max-pool 1
        self.pool1 = nn.Conv3d(in_channels=32, #Max pooling implemented as a convolution. Allows to have different shapes
                               out_channels=32,
                               kernel_size=2,
                               stride=2)
        # conv2 down
        self.conv2 = nn.Conv3d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        # max-pool 2
        self.pool2 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=2,
                               stride=2)
        # conv3 down
        self.conv3 = nn.Conv3d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        # max-pool 3
        self.pool3 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=2,
                               stride=2)
        # conv4 down (latent space)
        self.conv4 = nn.Conv3d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               padding=1)

        self.convGRU = ConvGRU3D(input_size=(4,4,4),
                        input_dim=128,
                        hidden_dim=[256],
                        kernel_size=(3,3,3),
                        num_layers=1,
                        dtype=torch.cuda.FloatTensor,
                        batch_first=True,
                        bias = True,
                        return_all_layers = False)


        # up-sample conv4
        self.up1 = nn.ConvTranspose3d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=2,
                                      stride=2)        
        # conv 5 (add up1 + conv3)
        self.conv5 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        # up-sample conv5
        self.up2 = nn.ConvTranspose3d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=2,
                                      stride=2)
        # conv6 (add up2 + conv2) 
        self.conv6 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        # up 3
        self.up3 = nn.ConvTranspose3d(in_channels=64,
                                      out_channels=32,
                                      kernel_size=2,
                                      stride=2)
        # conv7 (add up3 + conv1)
        self.conv7 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        # conv8 (classification)
        self.conv8 = nn.Conv3d(in_channels=32,
                               out_channels=output_size,
                               kernel_size=1)

    def wrapper(self, the_input, layer, layer_type = "normal"):
        num_time_steps = the_input.size(1)

        if layer_type == "normal":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, layer.out_channels, the_input.size(-3), the_input.size(-2), the_input.size(-1))).cuda()
        elif layer_type == "up":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, layer.out_channels, int(2*the_input.size(-3)), int(2*the_input.size(-2)), int(2*the_input.size(-1)))).cuda()
        else: # Pooling
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, the_input.size(2), the_input.size(-3) //2, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()

        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:,:] = F.relu(layer(the_input[:,i_tp,:,:,:,:]))
        return the_output




    def forward(self, x):

        # encoder
        x1 = self.wrapper(x, self.conv1)
        x1p = self.wrapper(x1, self.pool1, layer_type="pool")
        x2 = self.wrapper(x1p, self.conv2)
        x2p = self.wrapper(x2, self.pool2, layer_type="pool")
        x3 = self.wrapper(x2p, self.conv3)
        x3p = self.wrapper(x3, self.pool3, layer_type="pool")
        
        # latent space
        #x4 = F.relu(self.conv4(x3p))
        x4 = self.convGRU(x3p)[0][0]
        #x4 = self.wrapper(x3p, self.conv4)
        # decoder
        up1 = self.wrapper(x4, self.up1, layer_type="up")
        x5 = self.wrapper(up1+x3, self.conv5) # look how layers are added :o
        up2 = self.wrapper(x5, self.up2, layer_type="up")
        x6 = self.wrapper(up2 + x2, self.conv6)
        up3 = self.wrapper(x6, self.up3, layer_type="up")
        x7 = self.wrapper(up3 + x1, self.conv7)
        

        # output layer (2 classes)
        # we use a softmax layer to return probabilities for each class
        #out = F.softmax(self.conv8(x7), dim=1) #Dim 1 where to do the softmax. I have output [16 (batches),2 (multimodal),32x32x32 (patch size)], so I'm telling to do the softmax in the (2). For
        
        x7 = x7[:,-1,:,:,:,:]
        out = F.softmax(self.conv8(x7), dim=1)
        
        #For the project I will have [16, 4, 32x32x32] because I have 4 classes (background, CSF, WM, GM)
        return out



class UNet_ConvGRU_3D_alt(nn.Module):
    """
    Basic U-net model
    Changes: 
        Blocks implemented according to class blocks defined in unet3d file
    """

    def __init__(self, n_channels, n_classes, bilinear=True):

        super(UNet_ConvGRU_3D_alt, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 32)
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        self.convLSTM1 = ConvLSTM3D(input_size=(4,4,4),
                                input_dim=256,
                                hidden_dim=[256],
                                kernel_size=(3,3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.down4 = Down3D(256, 256)
        self.up1 = Up3D(512, 128, bilinear)
        self.up2 = Up3D(256, 64, bilinear)
        self.up3 = Up3D(128, 32, bilinear)
        self.up4 = Up3D(64, 32, bilinear)
        self.convLSTM2 = ConvLSTM3D(input_size=(32,32,32),
                                input_dim=32,
                                hidden_dim=[32],
                                kernel_size=(3,3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.outc = OutConv3D(32, n_classes)



        # Define wrappers
    def wrapper_conv(self, the_input, layer, out_channels, layer_type= "Down"):
        num_time_steps = the_input.size(1)
        if layer_type == "DoubleConv":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-3), the_input.size(-2), the_input.size(-1))).cuda()
        elif layer_type == "Down":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-3)//2, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()
        else: #layer_type == OutConv
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), out_channels, the_input.size(-3), the_input.size(-2), the_input.size(-1))).cuda()
            the_output = layer(the_input)
            return the_output
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:,:] = layer(the_input[:,i_tp,:,:,:,:])
        return the_output

    def wrapper_up(self, the_input1, the_input2, layer, out_channels):

        num_time_steps = the_input1.size(1)
        the_output = torch.zeros_like(torch.Tensor(the_input1.size(0), num_time_steps, out_channels, int(2*the_input1.size(-3)), int(2*the_input1.size(-2)), int(2*the_input1.size(-1)))).cuda()
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:,:] = layer(the_input1[:,i_tp,:,:,:,:], the_input2[:,i_tp,:,:,:,:])
        return the_output

    def forward(self, x):
        #x eg (5,3,2,32,32,32)
        x1 = self.wrapper_conv(x, self.inc, 32, layer_type = "DoubleConv") # (5,3,32,32,32,32)
        x2 = self.wrapper_conv(x1, self.down1, 64) # (5,3,64,16,16,16)
        x3 = self.wrapper_conv(x2, self.down2, 128) # (5,3,128,8,8,8)
        x4 = self.wrapper_conv(x3, self.down3, 256) # (5,3,256,4,4,4)
        x5_ = self.convLSTM1(x4)[0] # (5,3,256,4,4,4)
        x5 = self.wrapper_conv(x5_, self.down4, 256) # (5,3,256,2,2,2)
        #
        x = self.wrapper_up(x5, x4, self.up1,128) # (5,3,128,4,4,4)
        x = self.wrapper_up(x, x3, self.up2, 64) # (5,3,64,8,8,8)
        x = self.wrapper_up(x, x2, self.up3, 32) # (5,3,32,16,16,16)
        x = self.wrapper_up(x, x1, self.up4, 32) # (5,3,32,32,32,32)
        x = self.convLSTM2(x)[0]
        #x = self.convGRU2(x)[0][0].permute(0,2,1,3,4,5) #(5,32,3,32,32,32)

        x = x[:,-1,:,:,:,:] # (5,32,32,32,32)
        logits = F.softmax(self.wrapper_conv(x, self.outc, self.n_classes, layer_type="OutConv"), dim=1)
        return logits # (5,2,32,32,32)