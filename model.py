from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from network.modules import *
from network.prototype import *

class PrototypeModel(nn.Module):
    def __init__(self, n_prototypes, latent_size, n_classes):
        """
        Initializes the entire model -> Encoder, Decoder, and prototype network
        TODO: Possibly generalize this where you can give the Encoder and Decoder as arguments somehow?
        """
        super().__init__()

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        self.prototype = PrototypeClassifier(n_prototypes, latent_size, n_classes)

    def forward(self, x):
        """
        Performs one forward pass of the full model
        """
        encoded   = self.encoder.forward(x)         # f(x)
        decoded   = self.decoder.forward(encoded)   # g(f(x))
        prototype = self.prototype.forward(encoded) # h(f(x))

        return encoded, decoded, prototype
