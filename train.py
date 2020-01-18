import numpy as np
import os
import torch 
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.functional import one_hot
from torchvision.utils import save_image
from preprocessing import batch_elastic_transform
from model import PrototypeModel


# Global parameters for device and reproducibility
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globals
learning_rate = 0.0001
training_epoch = 1500
batch_size = 250

sigma = 4
alpha = 20
n_prototypes = 15
latent_size = 40
n_classes = 10

lambda_class = 20
lambda_ae = 1
lambda_1 = 1              # 1 and 2 here corresponds to the notation we used in the paper
lambda_2 = 1

model_path = 'models/'
prototype_path = 'images/prototypes/'
decoding_path = 'images/decoding/'

def run_epoch(model, dataloader, optimizer, iteration, epoch_loss, epoch_accuracy):
    for i, (images, labels) in enumerate(dataloader):
        # Up the iteration by 1
        iteration += 1

        # Transform images, then port to GPU
        images = batch_elastic_transform(images, sigma, alpha, 28, 28)
        images = images.to(device)
        labels = labels.to(device)
        oh_labels = one_hot(labels)

        # Forward pass
        _, decoding, (r1, r2, c) = model.forward(images)

        # Calculate loss: Crossentropy + Reconstruction + R1 + R2 
        # Crossentropy h(f(x)) and y
        ce = nn.CrossEntropyLoss()
        # reconstruction error g(f(x)) and x
        subtr = (decoding - images).view(-1, 28*28)
        re = torch.mean(torch.norm(subtr, dim=1))
        
        # Paper does 20 * ce and lambda_n = 1 for each regularization term
        # Calculate loss and get accuracy etc.
        loss = lambda_class * ce(c, torch.argmax(oh_labels, dim=1)) + lambda_ae * r1 + lambda_1 * r2 + lambda_2 * re
        #print( r1, r2)
        epoch_loss += loss.item()
        preds = torch.argmax(c,dim=1)
        corr = torch.sum(torch.eq(preds,labels))
        size = labels.shape[0]
        epoch_accuracy += corr.item()/size

        # Do backward pass and ADAM steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return iteration, epoch_loss, epoch_accuracy, decoding

def save_images(prototype_path, decoding_path, prototypes, decoding, epoch):
    if not os.path.exists(prototype_path):
            os.makedirs(prototype_path)
    if not os.path.exists(decoding_path):
        os.makedirs(decoding_path)
    save_image(prototypes, prototype_path+'prot{}.png'.format(epoch), nrow=5, normalize=True)
    save_image(decoding, decoding_path+'dec{}.png'.format(epoch), nrow=5, normalize=True)


def train_MNIST(learning_rate=0.0001, training_epochs=1500, batch_size=250, sigma=4, alpha=20):
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
        
        it, epoch_loss, epoch_acc, dec = run_epoch(proto, dataloader, optim, it, epoch_loss, epoch_acc)

        print(20*ce(c, torch.argmax(oh_labels, dim=1)), r1, r2, re)

        # Get prototypes and decode them to display
        prototypes = proto.prototype.get_prototypes()
        prototypes = prototypes.view(-1, 10, 2, 2)
        imgs = proto.decoder(prototypes)

        # Save images
        save_images(prototype_path, decoding_path, imgs, dec, epoch)

        # Save model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(proto, model_path+"proto.pth")

        # Print statement to check on progress
        print("Epoch: ", epoch, "Loss: ", epoch_loss / it, "Acc: ", epoch_acc/it)

#train_MNIST()