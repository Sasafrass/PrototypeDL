from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from network.modules import *
from network.prototype import *

class Model(object):
    #def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    def __init__(self):
        """
        Initializes the entire model -> Encoder, Decoder, and prototype network
        """
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        #self.prototype

    def forward(self, x):
        """
        Performs one forward pass of the full model
        """
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        # prototype = self.prototype.forward(encoded)

        return encoded, decoded #,prototype