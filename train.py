import numpy as np
import torch 
import torch.nn as nn
from model import PrototypeModel

images = torch.randn(100, 1, 28, 28)
labels = torch.randint(9, (100,))
m = PrototypeModel(14, 40, 10)
enc, dec, (p, c) = m.forward(images)
print(enc.shape, dec.shape, p.shape, c.shape)


# Calculate loss: Crossentropy + Reconstruction + R1 + R2 
# Crossentropy h(f(x)) and y
ce = nn.CrossEntropyLoss()
# reconstruction error g(f(x)) and x
re = torch.mean(torch.norm(dec - images) ** 2)
# regularization r1: Be close to at least one training example (get min distance to each datapoint=dimension 0)
r1 = torch.mean(torch.min(p, axis=0)[0])
# regularization r2: Be close to at least one prototype (get min distance to each prototype=dimension 1)
r2 = torch.mean(torch.min(p, axis=1)[0])

loss = 20*ce(c, labels) + re + r1 + r2
print(loss)
