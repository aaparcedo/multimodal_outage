import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Hyperparameters

image_dimension = 128
batch_size = 4
n_counties = 67
feature_vector_size = 8

# Model Classes

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(inplace=True),                                                     
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        encoder_input = []
        
        for batch in range(batch_size):
        
            x1 = self.inc(input[batch])      
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
        encoder_input = encoder_input.view(batch_size, n_counties, -1)
        
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
        self.fc7 = nn.Linear(16, 8)

    def forward(self, input):
        wave_net_input = []
        
        for batch in range(batch_size):
            x = torch.relu(self.fc1(input[batch]))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            x = torch.relu(self.fc6(x))
            x = self.fc7(x)
            wave_net_input.append(x)
            
        wave_net_input = torch.stack(wave_net_input)
        return wave_net_input
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.downsized_image_dimension = int(image_dimension / 16)
        self.output_layer_size = int(self.downsized_image_dimension * self.downsized_image_dimension * 64)
        self.fc1 = nn.Linear(feature_vector_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, self.output_layer_size) 
        
    def forward(self, input):
        expansion_input = []
        
        for batch in range(batch_size):
            x = torch.relu(self.fc1(input[batch]))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
            expansion_input.append(x)
            
        expansion_input = torch.stack(expansion_input)
        expansion_input = expansion_input.view(batch_size, n_counties, 64, self.downsized_image_dimension, self.downsized_image_dimension)
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
        
        for batch in range(batch_size):
            x = self.up1(input[batch], feature_maps[-1][feature_map_iteration])
            x = self.up2(x, feature_maps[-2][feature_map_iteration])
            x = self.up3(x, feature_maps[-3][feature_map_iteration])
            x = self.up4(x, feature_maps[-4][feature_map_iteration])
            logits = self.outc(x)
            predictions.append(logits)
            feature_map_iteration += 1
            
        predictions = torch.stack(predictions)
            
        return predictions
    
class Modified_UNET(nn.Module): 
    def __init__(self,input_channels=3, output_channels=3):
        super(Modified_UNET, self).__init__()
        self.contraction = Contraction(input_channels)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.expansion = Expansion(output_channels)
        
    def forward(self, input): 
        output = self.contraction(input)        
        output = self.encoder(output)
        output = self.decoder(output)
        feature_maps = self.contraction.feature_maps
        predicted_results = self.expansion(output, feature_maps)
        return predicted_results