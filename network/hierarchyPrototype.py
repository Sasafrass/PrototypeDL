import numpy as np
import torch
import torch.nn as nn
from helper import list_of_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HierarchyPrototypeClassifier(nn.Module):

    def __init__(self, n_sup_prototypes, latent_size, output_size, n_sub_prototypes):
        """
        Initialize prototype module with n_sup_prototypes super prototypes in space latent_size.
        Map to output_size classes.
        """
        super().__init__()
        self.n_sup_prototypes = n_sup_prototypes
        self.n_sub_prototpes = n_sub_prototypes
        self.latent_size = latent_size
        self.output_size = output_size
        # initialize n_sup_prototypes super prototypes, they are of size latent_size
        self.sup_prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_sup_prototypes, latent_size)))
        #self.sub_prototypes, self.linear_layers = self._createSubprototypes(output_size, n_prototypes, n_sub_prototypes, latent_size)
        self.sub_prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_sub_prototypes, latent_size)))
        #self.sub_prototypes = nn.Parameter(torch.zeros(n_sub_prototypes, latent_size)).to(device)

        # Linear layers for super prototypes and sub prototypes
        self.linear1 = nn.Linear(n_sup_prototypes, output_size)
        self.linear2 = nn.Linear(n_sub_prototypes, output_size)


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
                x : matrix of distances in latent space of shape (batch_size, n_sup_prototypes)
                out : non-normalized logits for every data point 
                      in the batch, shape (batch_size, output_size)
        """
        # Port input to float tensor
        input = input.float()
        # Latent space is 10x2x2 = 40
        input = input.view(len(input), self.latent_size)

        # Distances between input and prototypes
        sup_input_dist = list_of_distances(input, self.sup_prototypes)
        sub_input_dist = list_of_distances(input, self.sub_prototypes)

        #Compute unnormalized classification probabilities
        sup_out = self.linear1(sup_input_dist)
        sub_out = self.linear2(sub_input_dist)

        # Clone and detach so r3 and r4 do not effect sub_prototype parameters
        sub_clones = self.sub_prototypes.clone()
        sub_clones = sub_clones.detach()
        sub_clones.requires_grad = False
        # Calculate distances for sub- and super prototypes
        super_sub_dist = list_of_distances(self.sup_prototypes, sub_clones)

        # TODO: DIM VS AXIS: DOES IT MATTER?
        # r1 forces sub proto to be close to one training example
        # r2 forces one training example to be close to sub proto
        r1 = torch.mean(torch.min(sub_input_dist, axis = 0).values)
        r2 = torch.mean(torch.min(sub_input_dist, axis = 1).values)

        # Forcing sub prototype to look like super prototype and vice versa
        # r3 forces super prototype to be close to sub prototype
        # r4 forces sub prototype to be close to super prototype
        r3 = torch.mean(torch.min(super_sub_dist, axis = 1).values)
        r4 = torch.mean(torch.min(super_sub_dist, axis = 0).values)

        return sub_out, sup_out, r1, r2, r3, r4

    def get_prototypes(self):
        return self.sup_prototypes

    def get_sub_prototypes(self):
        return self.sub_prototypes