import numpy as np
import torch 
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.functional import one_hot

from model import PrototypeModel

def train_MNIST(learning_rate=0.002, training_epochs=10, batch_size=250):
    # Load data
    train_data = MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                            ]))
    test_data = MNIST('./data', train=False, download=True,transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                            ]))

    ### Initialize the model and the optimizer.
    proto = PrototypeModel(14, 40, 10)
    optim = torch.optim.Adam(proto.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_data, batch_size=batch_size)

    it = 0
    for epoch in range(training_epochs):
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            # TODO: Warp image data first.
            it += 1
            labels = one_hot(labels)
            _, dec, (p, c) = proto.forward(images)

            # Calculate loss: Crossentropy + Reconstruction + R1 + R2 
            # Crossentropy h(f(x)) and y
            ce = nn.CrossEntropyLoss()
            # reconstruction error g(f(x)) and x
            re = torch.mean(torch.norm(dec - images) ** 2)
            # regularization r1: Be close to at least one training example 
            # (get min distance to each datapoint=dimension 0)
            r1 = torch.mean(torch.min(p, axis=0)[0])
            # regularization r2: Be close to at least one prototype 
            # (get min distance to each prototype=dimension 1)
            r2 = torch.mean(torch.min(p, axis=1)[0])

            # Paper does 20 * ce and lambda_n = 1 for each regularization term
            loss = 20*ce(c, torch.argmax(labels, dim=1)) + re + r1 + r2
            # print(loss)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()
            optim.zero_grad()
            
        print(epoch, epoch_loss / it)
