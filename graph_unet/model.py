import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(8192, 512)
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
    def __init__(self, input_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 10000)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Instantiate the encoder
encoder = Encoder()

# random tensor
tensor_size = (8192, 1)
flattened_feature_map = torch.randn(tensor_size)
flattened_feature_map = flattened_feature_map.view(-1)
reduced_vector = encoder(flattened_feature_map.unsqueeze(0))

# Instantiate the decoder here because it needs the input size of the reduced vector
input_size = reduced_vector.size(1)
decoder = Decoder(input_size)
print("Reduced vector shape:", reduced_vector.shape)

'''
pass in reduced vector into GNN and get the output and store it in reduced_vector
'''

reconstructed_vector = decoder(reduced_vector)
print("Reconstructed vector shape:", reconstructed_vector.shape)

encoder_file_path = "encoder_checkpoint/encoder_weights.pth"
decoder_file_path = "decoder_checkpoint/decoder_weights.pth"

torch.save(encoder.state_dict(), encoder_file_path)
print("Encoder weights saved successfully.")

torch.save(decoder.state_dict(), decoder_file_path)
print("Decoder weights saved successfully.")
