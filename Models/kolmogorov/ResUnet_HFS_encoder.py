import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

device = torch.device('cuda')

class AdaptiveSwish(nn.Module):
    def __init__(self,beta_init=1.0):
        super(AdaptiveSwish,self).__init__()

        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)

class AdaptiveTanh(nn.Module):
    def __init__(self,alpha_init=1.0):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.parameter(torch.tensor(alpha_init))

    def forward(self,x):
        return torch.tanh(self.alpha*x)

class Rowdy(nn.Module):
    def __init__(self, beta_init=1.0, cos_terms=2):
        super(Rowdy, self).__init__()
        self.amplitudes = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(cos_terms)])
        self.frequencies = nn.ParameterList([nn.Parameter(0.1*torch.ones(1)) for _ in range(cos_terms)])

        self.base_frequencies = torch.arange(10, 10*(cos_terms+1), 10, dtype=torch.float32)

        self.beta = nn.Parameter(torch.tensor(beta_init))

def get_activation(activation_name):
    if activation_name =='adaptive_swish':
        return AdaptiveSwish()
    if activation_name =='adaptive_tanh':
        return AdaptiveTanh()
    if activation_name =='Rowdy':
        return Rowdy()
    if activation_name =='GELU':
        return nn.GELU(approximate='tanh')

class featscale(nn.Module):
    def __init__(self, patch_size,channels):
        super(featscale, self).__init__()
        self.patch_size = patch_size
        self.lambda1 = nn.Parameter(1.0*torch.ones(1,channels,1,1,1))
        self.lambda2 = nn.Parameter(1.0*torch.ones(1,channels,1,1,1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reshape into patches of shape (batch_size, channels, num_patches, patch_size, patch_size)
        X_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        num_patches = (height//self.patch_size)* (width//self.patch_size)
        X_patches = X_patches.reshape(batch_size, channels, num_patches, self.patch_size, self.patch_size)
        X_mean_patch = X_patches.mean(dim=2)
        X_mean_expanded = X_mean_patch.unsqueeze(2).expand(-1, -1, num_patches, -1, -1)
        
        #Generate X_d and X_h
        X_d = X_mean_expanded
        X_h = X_patches - X_d

        # #Combine X_d and X_h
        X = X_patches + self.lambda1*X_d + self.lambda2*X_h
        X = X.reshape(batch_size, channels, height//self.patch_size, width//self.patch_size, self.patch_size, self.patch_size)
        X = X.permute(0,1,2,4,3,5).reshape(batch_size, channels, height, width)
        return X

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
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
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock2, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            # nn.GELU(approximate='tanh'),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            # nn.GELU(approximate='tanh')
            activation
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
    def __init__(self, in_c,out_c, features = [64,128,256,512,512],bottleneck_feature=1024, patch_size_enc = [16,8,4,2,1], patch_size_dec=[16,8,4,2,1],activation_name='GELU'):
        super(ResUNet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lamb1_history = []
        self.lamb2_history = []
        self.activation = get_activation(activation_name)

        #Encoder and featscale
        self.encoder = nn.ModuleList()
        self.featscale = nn.ModuleList()
        self.featscale2 = nn.ModuleList()
        self.featscale3 = nn.ModuleList()

        num_layers = len(features)  # Assuming featscale lists match encoder layers
        self.w1 = nn.Parameter(0.1*torch.ones(num_layers))
        self.w2 = nn.Parameter(0.1*torch.ones(num_layers))
        self.w3 = nn.Parameter(0.1*torch.ones(num_layers))

        for i,feature in enumerate(features):
            self.encoder.append(ResidualBlock2(in_c,feature,self.activation))
            self.featscale.append(featscale(patch_size_enc[i],feature))

            self.featscale2.append(featscale(max(patch_size_enc[i]//2,1),feature))
            self.featscale3.append(featscale(max(patch_size_enc[i]//4,1),feature))

            in_c = feature

        # Bottleneck layer
        self.bottleneck = ResidualBlock2(features[-1], bottleneck_feature,self.activation)
        self.fs_bottleneck = featscale(patch_size=1, channels=bottleneck_feature)

        #Upsample and Decoder and featscale
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.featscale_up = nn.ModuleList()
        self.featscale_up2 = nn.ModuleList()
        self.featscale_up3 = nn.ModuleList()

        for i, feature in enumerate(reversed(features)):
            self.upsample.append(
                nn.ConvTranspose2d(bottleneck_feature, bottleneck_feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock(bottleneck_feature+feature, feature,self.activation))
            bottleneck_feature = feature

            self.featscale_up.append(featscale(patch_size_dec[-i-1],feature))
            self.featscale_up2.append(featscale(max(patch_size_dec[-i-1]//2,1),feature))
            self.featscale_up3.append(featscale(max(patch_size_dec[-i-1]//4,1),feature))
        
        self.final_conv = nn.Conv2d(features[0],self.out_c,kernel_size=1)

    def save_lambdas(self):
        self.lamb1_history.append(self.lamb1.item())
        self.lamb2_history.append(self.lamb2.item())

    def forward(self, x):
        #Downsampling path
        skip_connections = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            x1 = self.featscale[i](x)
            x2 = self.featscale2[i](x)
            x3 = self.featscale3[i](x)
            x = self.w1[i]*x1 + self.w2[i]*x2 + self.w3[i]*x3
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
        out = self.final_conv(x)

        return out
       
    def set_input(input_data):
        x = input_data[:,0:20,:,:]
        y = input_data[:,20:25,:,:]
        x = x.to(device)
        y = y.to(device)
        return x, y