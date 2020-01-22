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
        self.n_sub_prototpes = n_sub_prototypes
        self.latent_size = latent_size
        self.output_size = output_size
        # initialize n_prototypes prototypes, they are of size latent_size
        self.prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_prototypes, latent_size))).to(device)
        #self.sub_prototypes, self.linear_layers = self._createSubprototypes(output_size, n_prototypes, n_sub_prototypes, latent_size)
        self.sub_prototypes = nn.Parameter(torch.nn.init.uniform_(torch.zeros(n_sub_prototypes, latent_size))).to(device)
        #self.sub_prototypes = nn.Parameter(torch.zeros(n_sub_prototypes, latent_size)).to(device)

        # Linear layers for super prototypes and sub prototypes
        self.linear1 = nn.Linear(n_prototypes, output_size)
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
                x : matrix of distances in latent space of shape (batch_size, n_prototypes)
                out : non-normalized logits for every data point 
                      in the batch, shape (batch_size, output_size)
        """
        # Port input to float tensor
        input = input.float()
        # Latent space is 10x2x2 = 40
        input = input.view(len(input), self.latent_size)

        # Distances between input and prototypes
        x = torch.zeros((len(input), len(self.prototypes))).to(device)
        x = list_of_distances(input, self.prototypes)
        closest_index = torch.min(x, axis=0)
        
        # Terms r1, r2
        # regularization r1: Be close to at least one training example 
        # (get min distance to each datapoint=dimension 0)
        r1 = 0 #torch.mean(closest_index.values)

        # regularization r2: Be close to at least one superprototype 
        # (get min distance to each prototype=dimension 1)
        r2 = 0 #torch.mean(torch.min(x, axis=1).values)

        #compute the last prototype passes
        prototype_index = closest_index.indices
        out = self.linear1(x)

        # All sub prototype stuff here
        # Forcing sub prototype to look like input
        sub_input_dist = torch.zeros((len(input), len(self.sub_prototypes))).to(device)
        sub_input_dist = list_of_distances(input, self.sub_prototypes)

        # TODO: DIM VS AXIS: DOES IT MATTER?
        # r3 forces sub proto to be close to one training example
        # r4 forces one training example to be close to sub proto
        r3 = torch.mean(torch.min(sub_input_dist, axis = 0).values)
        r4 = torch.mean(torch.min(sub_input_dist, axis = 1).values)

        
        # Clone and detach so r5 and r6 do not effect sub_prototype parameters
        sub_prototype_input = self.sub_prototypes.clone().detach()
        sub_prototype_input.requires_grad = False
        # Calculate distances for sub- and super prototypes
        sub_super_dist = list_of_distances(self.prototypes, sub_prototype_input)

        # Forcing sub prototype to look like super prototype and vice versa
        # r5 forces super prototype to be close to sub prototype
        # r6 forces sub prototype to be close to super prototype
        r5 = torch.mean(torch.min(sub_super_dist, axis = 1).values)
        r6 = torch.mean(torch.min(sub_super_dist, axis = 0).values)
        # Last forward pass of sub prototypes
        sub_out = self.linear2(sub_input_dist)
    
        return r1, r2, out, r3, r4, r5, r6, sub_out

    def get_prototypes(self):
        return self.prototypes

    def get_sub_prototypes(self):
        return self.sub_prototypes