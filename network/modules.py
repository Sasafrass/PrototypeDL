import numpy as np
import torch
import torch.nn as nn

# Port device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Encoder class for convolutional neural network
class ConvEncoder(nn.Module):

    def __init__(self, latent_size):
        super().__init__()

        # Apparently padding=1 was necessary to get the same dimensions as listed in the paper.
        # There should be a way to do this automatically, which is what tensorflow probably does 
        # Although in the paper "zero padding" is used, which is kinda ambiguous in itself 
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, int(latent_size/4), kernel_size = 3, stride = 2, padding=1),
            nn.Sigmoid())

    def forward(self, input):
        """
        Perform forward pass of CNN encoder
        Args:
            Input:
                input : MNIST sized image of shape (batch_size, 1, 28, 28)

            Output:
                out : Encoded data of shape  (batch_size * 10 * 2 * 2)
        """
        # Perform full pass through network
        out = self.convnet(input)
        return out

class ConvDecoder(nn.Module):

    def __init__(self, latent_size):
        super().__init__()

        self.de1 = nn.ConvTranspose2d(int(latent_size/4),32,kernel_size=3, stride=2,padding=1)
        self.de2 = nn.ConvTranspose2d(32,32,kernel_size=3, stride=2,padding=1)
        self.de3 = nn.ConvTranspose2d(32,32,kernel_size=3, stride=2,padding=1)
        self.de4 = nn.ConvTranspose2d(32,1, kernel_size=3, stride=2,padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        """
        Perform forward pass of the CNN decoder
        Args:
            Input: 
                input : Encoded data of shape (batch_size * 10 * 2 * 2)
        
            Output:
                out : Decoded data of shape (batch_size * 1 * 28 * 28)
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
