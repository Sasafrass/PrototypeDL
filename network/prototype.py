import numpy as np
import torch
import torch.nn as nn
from helper import list_of_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # Latent space is 10x2x2 = 40
        input = input.view(len(input), self.latent_size)
        x = list_of_distances(input, self.prototypes)
        out = self.linear1(x)
        # regularization r1: Be close to at least one training example 
        # (get min distance to each datapoint=dimension 0)
        min1 = torch.mean(torch.min(x, axis=0).values)
        # regularization r2: Be close to at least one prototype 
        # (get min distance to each prototype=dimension 1)
        min2 = torch.mean(torch.min(x, axis=1).values)
        print(self.prototypes[0])
        return min1, min2, out

    def get_prototypes(self):
        return self.prototypes