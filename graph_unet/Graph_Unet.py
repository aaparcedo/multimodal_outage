import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        
        self.feature_maps = []
        
    def feature_maps(self): 

        return self.feature_maps
    
    def flattened_size(self): 
        
        return self.flattened_size
    
    def original_shape(self): 
        
        return self.original_shape
    
    def forward(self, input):
        
        x1 = self.inc(input)
        self.feature_maps.append(x1)
        
        x2 = self.down1(x1)
        self.feature_maps.append(x2)

        x3 = self.down2(x2)
        self.feature_maps.append(x3)

        x4 = self.down3(x3)
        self.feature_maps.append(x4)

        x5 = self.down4(x4)
        batch, _, _, _ = x5.shape
        encoder_input = x5.view(batch, -1)
    
        return encoder_input

class Encoder(nn.Module):
    def __init__(self, image_dimension):
        super(Encoder, self).__init__()
        self.downsized_image_dimension = image_dimension / 16
        self.first_layer_size = int(self.downsized_image_dimension * self.downsized_image_dimension * 1024)
        self.fc1 = nn.Linear(self.first_layer_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, input_size, batch_size, image_dimension):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.downsized_image_dimension = int(image_dimension / 16)
        self.output_layer_size = int(self.downsized_image_dimension * self.downsized_image_dimension * 1024)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, self.output_layer_size) 
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(self.batch_size, 1024, self.downsized_image_dimension, self.downsized_image_dimension)
        return x
    
class Expansion(nn.Module):
    def __init__(self, output_channels): 
        super(Expansion, self).__init__()
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, output_channels))
        
    def forward(self, input, feature_maps): 
        x = self.up1(input, feature_maps[-1])
        x = self.up2(x, feature_maps[-2])
        x = self.up3(x, feature_maps[-3])
        x = self.up4(x, feature_maps[-4])
        logits = self.outc(x)
        return logits
    
class Modified_UNET(nn.Module): 
    def __init__(self, batch_size=67, input_channels=3, output_channels=3, feature_vector_size=8, image_dimension=128):
        super(Modified_UNET, self).__init__()
        self.contraction = Contraction(input_channels)
        self.encoder = Encoder(image_dimension)
        self.decoder = Decoder(feature_vector_size, batch_size, image_dimension)
        self.expansion = Expansion(output_channels)
        
    def forward(self, input): 
        output = self.contraction(input)
        output = self.encoder(output)
        output = self.decoder(output)        
        feature_maps = self.contraction.feature_maps
        predicted_results = self.expansion(output, feature_maps)
        return predicted_results