import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh')
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
            #nn.GELU(approximate='tanh')

        )
    def forward(self, x):
        shortcut = self.skip(x)
        out = self.residual(x)
        return out+shortcut

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            nn.GELU(approximate='tanh')
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(1,out_channels)
            #nn.GELU(approximate='tanh')

        )
    def forward(self, x):
        shortcut = self.skip(x)
        out = self.residual(x)
        return out+shortcut

class ResUNet(nn.Module):
    def __init__(self, in_c,out_c, features = [64,128,256,256,512],bottleneck_feature=512):
        super(ResUNet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        #Encoder
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(ResidualBlock2(in_c,feature))
            in_c = feature
        
        # Bottleneck layer
        self.bottleneck = ResidualBlock2(features[-1], bottleneck_feature)

        #Upsample and Decoder
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.upsample.append(
                nn.ConvTranspose2d(bottleneck_feature, bottleneck_feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock(bottleneck_feature+feature, feature))
            bottleneck_feature = feature
        
        self.final_conv = nn.Conv2d(features[0],self.out_c,kernel_size=1)

    def forward(self, x):
        #Downsampling path
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        x = self.bottleneck(x)

        #Upsampling path
        skip_connections = skip_connections[::-1]
        for up in range(len(self.decoder)):
            x = self.upsample[up](x)
            x = torch.cat((x, skip_connections[up]),dim=1)
            x = self.decoder[up](x)
        
        out = self.final_conv(x)

        return out
       
    def set_input(input_data):
        TV = torch.cat((input_data[1],input_data[2]),dim=1)
        TV = TV.float().to(device)
        T_next = input_data[3].float().to(device)
        return TV, T_next

    def set_input_TV(input_data):
        TV = torch.cat((input_data[1],input_data[2]),dim=1)
        TV = TV.float().to(device)
        T_next = input_data[3].float().to(device)
        return TV, T_next
        
    def set_input_kolmo(input_data):
        x = input_data[:,0:20,:,:]
        y = input_data[:,20:25,:,:]
        x = x.to(device)
        y = y.to(device)
        return x, y

