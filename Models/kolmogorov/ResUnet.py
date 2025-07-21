import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda')
torch.manual_seed(1234)
np.random.seed(1234)

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

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock2, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            activation,
            # nn.GELU(approximate='tanh'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            activation
            # nn.GELU(approximate='tanh')
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
    def __init__(self, in_c,out_c, features = [64,128,256,512,512],bottleneck_feature=1024,activation_name='GELU'):
        super(ResUNet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.activation = get_activation(activation_name)

        #Encoder
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(ResidualBlock2(in_c,feature,self.activation))
            in_c = feature
        
        # Bottleneck layer
        self.bottleneck = ResidualBlock2(features[-1], bottleneck_feature,self.activation)

        #Upsample and Decoder
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.upsample.append(
                nn.ConvTranspose2d(bottleneck_feature, bottleneck_feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock2(bottleneck_feature+feature, feature,self.activation))
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
        x = input_data[:,0:20,:,:]
        y = input_data[:,20:25,:,:]
        x = x.to(device)
        y = y.to(device)
        return x, y
    