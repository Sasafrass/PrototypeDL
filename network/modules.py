import numpy as np
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize Linear Layers
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size = 3, stride = 2),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2),
            nn.Conv2d(32, 10, kernel_size = 3, stride = 2)
        )


        self.linear1 = nn.Linear(784, hidden_dim)
        self.linearmu = nn.Linear(hidden_dim, z_dim)
        self.linearsig = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder for convolutional network.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ht = torch.nn.functional.tanh(self.linear1(input)).to(device)
        mean, std = self.linearmu(ht), self.linearsig(ht).to(device)

        return mean, std

class ConvDecoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        # Initialize linear layers
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        # First run one hidden layer, then get means output
        ht = torch.nn.functional.relu(self.linear1(input))
        mean = torch.sigmoid(self.linear2(ht))

        return mean
