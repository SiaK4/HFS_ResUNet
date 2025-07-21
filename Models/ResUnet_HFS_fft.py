import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

device = torch.device('cuda')

class HFS(nn.Module):
    def __init__(self,channels,cutoff):
        super(HFS, self).__init__()
        self.channels = channels
        self.cutoff = cutoff
        self.lambda1 = nn.Parameter(torch.ones(1,channels,1,1))
        self.lambda2 = nn.Parameter(torch.ones(1,channels,1,1))
    def decompose_frequency(self,x):
        rows, cols = x.shape[-2:]
        center_row, center_col = rows//2, cols//2

        mask_low = torch.zeros((rows, cols), dtype=torch.float32)
        mask_low[center_row-self.cutoff:center_row+self.cutoff, center_col-self.cutoff:center_col+self.cutoff] = 1
        mask_high = 1 - mask_low
        mask_low = mask_low[None, None, :, :]
        mask_high = mask_high[None, None, :, :]

        #Perform FFT
        fft = torch.fft.fft2(x, dim=(-2,-1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2,-1))

        # Apply masks
        fft_low = fft_shifted*mask_low.to(fft.device)
        fft_high = fft_shifted*mask_high.to(fft.device)

        # Inverse FFT to reconstruct images
        fft_low_ishifted = torch.fft.ifftshift(fft_low, dim=(-2,-1))
        fft_high_ishifted = torch.fft.ifftshift(fft_high, dim=(-2,-1))

        low = torch.fft.ifft2(fft_low_ishifted, dim=(-2,-1)).real
        high = torch.fft.ifft2(fft_high_ishifted, dim=(-2,-1)).real

        return low, high

    def forward(self,x):
        low, high = self.decompose_frequency(x)
        low = low*self.lambda1 
        high = high*self.lambda2

        x = low+high
        return x

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
    def __init__(self, in_c,out_c, features = [64,128,256,512,512],bottleneck_feature=1024, cutoff=[10,10,10,10,10], bottleneck_cutoff=1):
        super(ResUNet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        #Encoder and featscale
        self.encoder = nn.ModuleList()
        self.featscale = nn.ModuleList()
        for i,feature in enumerate(features):
            self.encoder.append(ResidualBlock2(in_c,feature))
            self.featscale.append(HFS(feature,cutoff[i]))
            in_c = feature

        # Bottleneck layer
        self.bottleneck = ResidualBlock2(features[-1], bottleneck_feature)
        self.fs_bottleneck = HFS(channels=bottleneck_feature, cutoff=bottleneck_cutoff)

        #Upsample and Decoder and featscale
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.featscale_up = nn.ModuleList()

        for i, feature in enumerate(reversed(features)):
            self.upsample.append(
                nn.ConvTranspose2d(bottleneck_feature, bottleneck_feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock(bottleneck_feature+feature, feature))
            bottleneck_feature = feature

            self.featscale_up.append(HFS(feature,cutoff[-i]))
        
        self.final_conv = nn.Conv2d(features[0],self.out_c,kernel_size=1)

    def forward(self, x):
        #Downsampling path
        skip_connections = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            x = self.featscale[i](x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        x = self.bottleneck(x)
        x = self.fs_bottleneck(x)

        #Upsampling path
        skip_connections = skip_connections[::-1]
        for up in range(len(self.decoder)):
            x = self.upsample[up](x)
            x = torch.cat((x, skip_connections[up]),dim=1)
            x = self.decoder[up](x)
            x = self.featscale_up[up](x)
        out = self.final_conv(x)

        return out
       
    def set_input(input_data):
        TV = torch.cat((input_data[1],input_data[2]),dim=1)
        TV = TV.float().to(device)
        T_next = input_data[3].float().to(device)
        return TV, T_next
