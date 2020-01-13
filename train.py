import numpy as np
import torch 
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.functional import one_hot
from torchvision.utils import save_image
from preprocessing import batch_elastic_transform

from model import PrototypeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_MNIST(learning_rate=0.002, training_epochs=10, batch_size=250, sigma=4, alpha=20):
    # Load data
    train_data = MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                            ]))
    test_data = MNIST('./data', train=False, download=True,transform=transforms.Compose([
                                                transforms.ToTensor(),
                                            ]))


    ### Initialize the model and the optimizer.
    proto = PrototypeModel(15, 40, 10).to(device)
    optim = torch.optim.Adam(proto.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_data, batch_size=batch_size)

    # Run for a number of epochs
    for epoch in range(training_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        it = 0
        for i, (images, labels) in enumerate(dataloader):

            # Up the iteration by 1
            it += 1

            # Transform images, then port to GPU
            images = batch_elastic_transform(images, sigma, alpha, 28, 28)
            images = images.to(device)
            labels = labels.to(device)

            # TODO: Warp image data first.
            oh_labels = one_hot(labels)
            _, dec, (r1, r2, c) = proto.forward(images)

            # Calculate loss: Crossentropy + Reconstruction + R1 + R2 
            # Crossentropy h(f(x)) and y
            ce = nn.CrossEntropyLoss()
            # reconstruction error g(f(x)) and x
            re = torch.mean(torch.norm(dec - images) ** 2)
            
            # Paper does 20 * ce and lambda_n = 1 for each regularization term
            # Calculate loss and get accuracy etc.
            loss = 20*ce(c, torch.argmax(oh_labels, dim=1)) + re + r1 + r2
            #print( r1, r2)
            epoch_loss += loss.item()
            preds = torch.argmax(c,dim=1)
            corr = torch.sum(torch.eq(preds,labels))
            size = labels.shape[0]
            epoch_acc += corr.item()/size

            # Do backward pass and ADAM steps
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Get prototypes and decode them to display
        prototypes = proto.prototype.get_prototypes()
        prototypes = prototypes.view(-1,10,2,2)
        imgs = proto.decoder(prototypes)

        # Save images
        save_image(imgs, 'images/prot{}.png'.format(epoch), nrow=5, normalize=True)
        save_image(dec, 'images/dec{}.png'.format(epoch), nrow=5, normalize=True)

        # Save model
        torch.save(proto, "models/proto.pth")

        # Print statement to check on progress
        print("Epoch: ", epoch, "Loss: ", epoch_loss / it, "Acc: ", epoch_acc/it)

train_MNIST()