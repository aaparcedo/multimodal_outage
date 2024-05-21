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

from .graph_wavenet import gwnet

# Hyperparameters

image_dimension = 128
batch_size = 4
n_counties = 67
n_timestep = 7
feature_vector_size = 16

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
        self.fc1 = nn.Linear(self.first_layer_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, feature_vector_size)

    def forward(self, input):
        wave_net_input = []

        for county in range(n_counties):
            x = torch.relu(self.fc1(input[county]))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            wave_net_input.append(x)


        wave_net_input = torch.stack(wave_net_input)
        return wave_net_input

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.downsized_image_dimension = int(image_dimension / 16)
        self.output_layer_size = int(self.downsized_image_dimension * self.downsized_image_dimension * 64)
        self.fc1 = nn.Linear(feature_vector_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, self.output_layer_size)

    def forward(self, input):
        expansion_input = []

        for county in range(n_counties):
            x = torch.relu(self.fc1(input[county]))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
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
        self.gwn = gwnet(device='cuda', num_nodes=67, dropout=0.3, supports=supports, in_dim=feature_vector_size, out_dim=feature_vector_size)

    def forward(self, input):
        result = []
        for batch in range(input.shape[0]):
            output = self.contraction(input[batch])
            output = self.encoder(output)
            output = output.unsqueeze(0).permute(0, 3, 1, 2)
            output = self.gwn(output)
            output = output.squeeze(0).permute(1, 2, 0)
            output = self.decoder(output)
            feature_maps = self.contraction.feature_maps
            predicted_results = self.expansion(output, feature_maps)
            result.append(predicted_results)
        result = torch.stack(result)
        return result
