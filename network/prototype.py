import numpy as np
import torch
import torch.nn as nn

class PrototypeClassifier(nn.Module):

    def __init__(self, n_prototypes, latent_size, output_size):
        """
        Initialize prototype module with n_prototypes prototypes in space latent_size.
        Map to output_size classes.
        """
        super().__init__()

        # initialize n_prototypes prototypes, they are of size latent_size
        self.prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_prototypes, latent_size)))

        # Initialize Linear Layer
        self.linear1 = nn.Linear(n_prototypes, output_size)


    def forward(self, input):
        """
        Perform forward pass for the prototype network.
        """
        input = input.float()
        # Latent space is 2x10x10 = 40
        input = input.view(len(input), 40)
        x = torch.zeros((len(input), len(self.prototypes)))
        for i, input_row in enumerate(input): # Todo: Remove loop
             x[i] = torch.sqrt(torch.sum((input_row - self.prototypes)**2))
             
        out = self.linear1(x)

        return x, out

