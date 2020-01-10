import numpy as np
import torch
import torch.nn as nn

# Port device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Encoder class for convolutional neural network
class ConvEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize the ConvNet
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size = 3, stride = 2),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2),
            nn.Sigmoid(),
            nn.Conv2d(32, 10, kernel_size = 3, stride = 2)
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder for convolutional network.
        """
        # Perform full pass through network
        out = self.convnet(input)

        return out

class ConvDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize deconvnet decoder
        self.deconvnet = nn.Sequential(
            nn.ConvTranspose2d(10, 32, kernel_size = 3, stride = 1),
            nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 1),
            nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 1), 
            nn.ConvTransPose2d(32, 1, kernel_size = 3, stride = 1)   
        )

    def forward(self, input):
        """
        Perform forward pass of decoder.
        """
        
        # Perform pass of decoder network
        out = self.deconvnet(input)

        return out
