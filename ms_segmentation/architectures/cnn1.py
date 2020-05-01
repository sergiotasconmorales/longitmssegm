import torch
import torch.nn as nn
import torch.nn.functional as F
from .c_lstm import ConvLSTM3D



class ConvAndPooling(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.simple_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CNN1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CNN1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.first_conv = ConvAndPooling(n_channels, 16)
        self.clstm = ConvLSTM3D(input_size=(7,7,7),
                                input_dim=16,
                                hidden_dim=[32],
                                kernel_size=(3,3,3),
                                num_layers=1,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False)
        self.second_conv = ConvAndPooling(32, 64)
        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(3*64*3*3*3),
            nn.Linear(3*64*3*3*3,256),
            nn.Linear(256, 2)
        )

        

    def wrapper_conv(self, the_input, layer, out_channels, layer_type= "Down"):
        num_time_steps = the_input.size(1)
        if layer_type == "ConvAndPooling":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-3)//2, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()
        else: #layer_type == OutConv
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), out_channels, the_input.size(-3), the_input.size(-2), the_input.size(-1))).cuda()
            the_output = layer(the_input)
            return the_output
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:,:] = layer(the_input[:,i_tp,:,:,:,:])
        return the_output

    def forward(self, x):

        # if x has size (5,3,3,15,15,15)
        x = self.wrapper_conv(x, self.first_conv, 16, layer_type = "ConvAndPooling") # (5,3,16,7,7,7)
        x = self.clstm(x)[0] #(5,3,32,7,7,7)
        x = self.wrapper_conv(x,self.second_conv, 64, layer_type = "ConvAndPooling") # (5,3,64,3,3,3)
        x = x.view(x[:,-1,:,:,:,:].size(0), -1) #(5,5184)
        logits = F.softmax(self.linear_layers(x), dim=1) #(5,2)
        return logits



class CNN2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CNN2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.first_conv = ConvAndPooling(n_channels, 32)
        self.second_conv = ConvAndPooling(32, 64)

        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Linear(256, 2)
        )

        

    def wrapper_conv(self, the_input, layer, out_channels, layer_type= "Down"):
        num_time_steps = the_input.size(1)
        if layer_type == "ConvAndPooling":
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), num_time_steps, out_channels, the_input.size(-3)//2, the_input.size(-2)//2, the_input.size(-1)//2)).cuda()
        else: #layer_type == OutConv
            the_output = torch.zeros_like(torch.Tensor(the_input.size(0), out_channels, the_input.size(-3), the_input.size(-2), the_input.size(-1))).cuda()
            the_output = layer(the_input)
            return the_output
        for i_tp in range(num_time_steps):
            the_output[:,i_tp,:,:,:,:] = layer(the_input[:,i_tp,:,:,:,:])
        return the_output

    def forward(self, x):

        # if x has size ()
        x = self.first_conv(x) # 5,32,7,7,7
        x = self.second_conv(x) # 5,64,3,3,3
        x = x.view(x.size(0), -1) # 5,1728
        logits = F.softmax(self.linear_layers(x), dim=1) # 5,2
        return logits