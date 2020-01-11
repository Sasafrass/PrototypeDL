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

        self.latent_size = latent_size
        # initialize n_prototypes prototypes, they are of size latent_size
        self.prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_prototypes, latent_size)))

        # Initialize Linear Layer
        # TODO: Special case where m=k, aka n_prototypes = output_size
        # Then simply set linear to negative identity matrix
        self.linear1 = nn.Linear(n_prototypes, output_size)


    def forward(self, input):
        """
        Perform forward pass for the prototype network.
        Args:
            Input:
                input : Latent space encodings of shape (batch_size, latent_size)
                        Where latent_size can be any amount of dimensions which multiply to latent_size

            Output:
                x : matrix of distances in latent space of shape (batch_size, n_prototypes)
                out : non-normalized logits for every data point 
                      in the batch, shape (batch_size, output_size)
        """
        input = input.float()
        # Latent space is 2x10x10 = 40
        input = input.view(len(input), 40)
        x = torch.zeros((len(input), len(self.prototypes)))
        for i, input_row in enumerate(input): # TODO: Remove loop
             x[i] = torch.sqrt(torch.sum((input_row - self.prototypes)**2))
        
        out = self.linear1(x)

        return x, out

