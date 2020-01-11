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
        # Apparently padding=1 was necessary to get the same dimensions as listed in the paper.
        # There should be a way to do this automatically, which is what tensorflow probably does 
        # Although in the paper "zero padding" is used, which is kinda ambiguous in itself 
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid()
            
        )
        self.mid = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid()
        )

        self.final = nn.Sequential(nn.Conv2d(32, 10, kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid())

    def forward(self, input):
        """
        Perform forward pass of encoder for convolutional network.
        """
        # Perform full pass through network
        out = self.convnet(input)
        print("Shape after first layer\t", out.shape)
        out = self.mid(out)
        print("Shape after middle layers\t", out.shape)
        out = self.final(out)
        print("Final shape\t", out.shape)
        return out

class ConvDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.de1 = nn.ConvTranspose2d(10,32,kernel_size=3, stride=2,padding=1)
        self.de2 = nn.ConvTranspose2d(32,32,kernel_size=3, stride=2,padding=1)
        self.de3 = nn.ConvTranspose2d(32,32,kernel_size=3, stride=2,padding=1)
        self.de4 = nn.ConvTranspose2d(32,1,kernel_size=3, stride=2,padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        """
        Perform forward pass of decoder.
        """
        # Output_size is necessary during convolution
        b = len(input)
        out = self.de1(input, output_size=(b,32,4,4))
        out = self.sigmoid(out)
        out = self.de2(out, output_size=(b,32,7,7))
        out = self.sigmoid(out)
        out = self.de3(out, output_size=(b,32,14,14))
        out = self.sigmoid(out)
        out = self.de4(out, output_size=(b,1,28,28))
        out = self.sigmoid(out)
        return out
