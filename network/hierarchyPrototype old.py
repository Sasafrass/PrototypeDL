import numpy as np
import torch
import torch.nn as nn
from helper import list_of_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HierarchyPrototypeClassifier(nn.Module):

    def __init__(self, n_prototypes, latent_size, output_size, n_sub_prototypes):
        """
        Initialize prototype module with n_prototypes prototypes in space latent_size.
        Map to output_size classes.
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.latent_size = latent_size
        self.output_size = output_size
        # initialize n_prototypes prototypes, they are of size latent_size
        self.prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_prototypes, latent_size))).to(device)
        self.sub_prototypes, self.linear_layers = self._createSubprototypes(output_size, n_prototypes, n_sub_prototypes, latent_size)


    def _createSubprototypes(self, output_size, n_prototypes, n_sub_prototypes, latent_size):
            sub_prototypes = np.empty(n_prototypes, dtype=object)
            linear_layers = np.empty(n_prototypes, dtype = object )
            for subset in range(n_prototypes):
                sub_prototypes[subset] = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_sub_prototypes, latent_size))).to(device)
                linear_layers[subset] = nn.Linear(n_sub_prototypes, output_size).to(device)

            return sub_prototypes, linear_layers

    def _compute_linear(self, input, index):
        input = input.to(device)
        x = list_of_distances(input, self.sub_prototypes[index])

        out = self.linear_layers[index].forward(x)

        # regularization r1: Be close to at least one training example 
        # (get min distance to each datapoint=dimension 0)
        sub_min1 = torch.mean(torch.min(x, axis=0).values)
        # regularization r2: Be close to at least one prototype 
        # (get min distance to each prototype=dimension 1)
        sub_min2 = torch.mean(torch.min(x, axis=1).values)

        return out, sub_min1, sub_min2

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
        x = torch.zeros((len(input), len(self.prototypes))).to(device)

        x = list_of_distances(input, self.prototypes)
        closest_index = torch.min(x, axis=1)
        
        # Terms r1, r2
        min1 = torch.mean(closest_index.values)
        min2 = torch.mean(torch.min(x, axis=0).values)

        #compute the sub prototypes
        prototype_index = closest_index.indices
        out = torch.zeros((len(input), self.output_size)).to(device)

        sub_min1 = 0
        sub_min2 = 0

        for ix in range(self.n_prototypes):
            rearrange_index = prototype_index == ix
            test = rearrange_index.float().sum()
            values = input[rearrange_index]
            if len(values) == 0: continue
            output, idx_sub_min1, idx_sub_min2 = self._compute_linear(values, ix)
            sub_min1 += idx_sub_min1 #/ test 
            sub_min2 += idx_sub_min2 #/ test
            out[rearrange_index] = output
        
        # terms r3, r4
        sub_min1 /= self.n_prototypes
        sub_min2 /= self.n_prototypes

        return min1, min2, sub_min1, sub_min2, out

    def get_prototypes(self):
        return self.prototypes

    def get_sub_prototypes(self):
        return self.sub_prototypes