import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Hyperparameters

image_dimension = 128
batch_size = 4
n_counties = 67
n_timestep = 7 
feature_vector_size = 8

# Graph WaveNet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=1,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, 1), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, 1),  dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,

                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field


    def forward(self, input):
        in_len = input.size(2)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

# End of Graph WaveNet model class(es)


# Model Classes: U-Net & Encoder-Decoder

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),                                                     
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)                                                       
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
                
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Contraction(nn.Module):
    def __init__(self, in_channels): 
        super().__init__()
        self.inc = (DoubleConv(in_channels, 4))
        self.down1 = (Down(4, 8))
        self.down2 = (Down(8, 16))
        self.down3 = (Down(16, 32))
        self.down4 = (Down(32, 64))
        self.feature_maps = [[] for _ in range(4)]
    
    def forward(self, input):
        self.feature_maps = [[] for _ in range(4)]
        encoder_input = []
        
        for county in range(n_counties):
        
            x1 = self.inc(input[county])      
            self.feature_maps[0].append(x1)            
            
            x2 = self.down1(x1)             
            self.feature_maps[1].append(x2)
            
            x3 = self.down2(x2)            
            self.feature_maps[2].append(x3)

            x4 = self.down3(x3)            
            self.feature_maps[3].append(x4)

            x5 = self.down4(x4)             
            encoder_input.append(x5)
            
        for feature_map in range(len(self.feature_maps)): 
            self.feature_maps[feature_map] = torch.stack(self.feature_maps[feature_map])
              
        encoder_input = torch.stack(encoder_input)
        encoder_input = encoder_input.view(n_counties, n_timestep, -1)
        
        return encoder_input
    
class Encoder(nn.Module): 
    def __init__(self):
        super(Encoder, self).__init__()
        self.downsized_image_dimension = image_dimension / 16
        self.first_layer_size = int(self.downsized_image_dimension * self.downsized_image_dimension * 64)
        self.fc1 = nn.Linear(self.first_layer_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8) # (16, 8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        wave_net_input = []
        
        for county in range(n_counties):
            x = torch.relu(self.fc1(input[county]))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout(x)
            x = torch.relu(self.fc4(x))
            x = self.dropout(x)
            x = torch.relu(self.fc5(x))
            x = self.dropout(x)
            x = torch.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.fc7(x)
            wave_net_input.append(x)
            
        wave_net_input = torch.stack(wave_net_input)
        return wave_net_input
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.downsized_image_dimension = int(image_dimension / 16)
        self.output_layer_size = int(self.downsized_image_dimension * self.downsized_image_dimension * 64)
        self.fc1 = nn.Linear(feature_vector_size, 64) # (, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, self.output_layer_size) 
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, input):
        expansion_input = []
        
        for county in range(n_counties):
            x = torch.relu(self.fc1(input[county]))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout(x)
            x = torch.relu(self.fc4(x))
            x = self.dropout(x)
            x = self.fc5(x)
            expansion_input.append(x)
            
        expansion_input = torch.stack(expansion_input)
        expansion_input = expansion_input.view(n_counties, n_timestep, 64, self.downsized_image_dimension, self.downsized_image_dimension)
        return expansion_input
    
class Expansion(nn.Module):  
    def __init__(self, output_channels): 
        super(Expansion, self).__init__()
        self.up1 = (Up(64, 32))
        self.up2 = (Up(32, 16))
        self.up3 = (Up(16, 8))
        self.up4 = (Up(8, 4))
        self.outc = (OutConv(4, output_channels))
        
    def forward(self, input, feature_maps):
        predictions = []
        feature_map_iteration = 0
        
        for county in range(n_counties):
            x = self.up1(input[county], feature_maps[-1][feature_map_iteration])
            x = self.up2(x, feature_maps[-2][feature_map_iteration])
            x = self.up3(x, feature_maps[-3][feature_map_iteration])
            x = self.up4(x, feature_maps[-4][feature_map_iteration])
            logits = self.outc(x)
            predictions.append(logits)
            feature_map_iteration += 1
            
        predictions = torch.stack(predictions)
            
        return predictions
    
class Modified_UNET(nn.Module): 
    def __init__(self,supports, input_channels=3, output_channels=3):
        super(Modified_UNET, self).__init__()
        self.contraction = Contraction(input_channels)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.expansion = Expansion(output_channels)

        # TODO: make dynamic
        self.gwn = gwnet(device='cuda', num_nodes=67, dropout=0.3, supports=supports, in_dim=8, out_dim=8)
 
    def forward(self, input):
        result = []  
        for batch in range(input.shape[0]):
            output = self.contraction(input[batch])            
            output = self.encoder(output)
            #print(f'encoder output shape: {output.shape}')
            
            output = output.unsqueeze(0).permute(0, 3, 1, 2)
            #print(f'encoder output shape unsqueeze and reshaped: {output.shape}')            

            output = self.gwn(output)
            #print(f"gwn output shape: {output.shape}")
            output = output.squeeze(0).permute(1, 2, 0)
            #print(f"output shape (squeeze and permute) before decoder: {output.shape}")
  
            output = self.decoder(output)            
            feature_maps = self.contraction.feature_maps
            predicted_results = self.expansion(output, feature_maps)
            result.append(predicted_results)
            
        result = torch.stack(result)
        return result
