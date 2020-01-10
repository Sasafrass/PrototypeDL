import numpy as np
import torch
import torch.nn as nn

class PrototypeClassifier(nn.Module):

    def __init__(self, n_prototypes, input_size, output_size):
        super().__init__()

        # initialize prototypes
        self.prototypes = nn.parameter(torch.nn.init.uniform_(torch.zeros(n_prototypes, output_size)))

        # Initialize Linear Layers
        self.linear1 = nn.Linear(n_prototypes, output_size)


    def forward(self, input):
        """
        Perform forward pass of encoder for convolutional network.
        """

        """
        TODO this needs to be optimized
        print(input)
        print(self.prototypes)
        input = input.float()
        x = input.unsqueeze(1) - self.prototypes
        x = x.reshape(-1, input.shape[1])
        print(x)
        x = torch.sqrt(x)
        """
        input = input.float()
        x = torch.zeros((len(input), len(self.prototypes)))
        for i, input_row in enumerate(input):
            for j, prot_row in enumerate(self.prototypes):
                x[i][j] = torch.sqrt(torch.sum((input_row - prot_row)**2))

        out = self.linear1(x)

        return out

