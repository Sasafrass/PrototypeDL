import numpy as np
import os
import torch 
import argparse
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.functional import one_hot
from torchvision.utils import save_image
from preprocessing import batch_elastic_transform
from model import PrototypeModel, HierarchyModel

# Global parameters for device and reproducibility
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=7,
                        help='seed for reproduction')
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'models/'
prototype_path = 'images/prototypes/'
decoding_path = 'images/decoding/'

# Training details
#learning_rate = 0.0001
#training_epoch = 1500
#batch_size = 250
#save_every = 50

# Warping parameters
#sigma = 4
#alpha = 20

# Model details 
#hierarchical = False
#n_prototypes = 15
#n_sub_prototypes = 3
#latent_size = 40
#n_classes = 10

# Loss weights for cross entropy, reconstruction loss and the two extra terms as described in the paper
lambda_class = 20
lambda_ae = 1
lambda_1 = 1
lambda_2 = 1

def run_epoch_n(sigma, alpha, model, dataloader, optimizer,
        iteration,epoch_loss, epoch_accuracy):
    
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

        crossentropy_loss = ce(c, torch.argmax(oh_labels, dim=1))
        loss = lambda_class * crossentropy_loss + lambda_ae * re + lambda_1 * r2 + lambda_2 * re
        
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
def run_epoch(hierarchical, sigma, alpha,       # Model parameters
        model, dataloader, optimizer,           # Training objects
        iteration, epoch_loss, epoch_accuracy): # Evaluation 
    for i, (images, labels) in enumerate(dataloader):
        # Up the iteration by 1
        iteration += 1

        # Transform images, then port to GPU
        images = batch_elastic_transform(images, sigma, alpha, 28, 28)
        images = images.to(device)
        labels = labels.to(device)
        oh_labels = one_hot(labels)

        # Forward pass
        if hierarchical:
            _, decoding, (r1, r2, r3, r4, c) = model.forward(images)
        else:
            _, decoding, (r1, r2, c) = model.forward(images)

        # Calculate loss: Crossentropy + Reconstruction + R1 + R2 
        # Crossentropy h(f(x)) and y
        ce = nn.CrossEntropyLoss()
        # reconstruction error g(f(x)) and x
        subtr = (decoding - images).view(-1, 28*28)
        re = torch.mean(torch.norm(subtr, dim=1))
        
        # Paper does 20 * ce and lambda_n = 1 for each regularization term
        # Calculate loss and get accuracy etc.

        crossentropy_loss = ce(c, torch.argmax(oh_labels, dim=1))
        if hierarchical:
            loss = lambda_class * crossentropy_loss + lambda_ae * re + \
                lambda_1 * r1 + lambda_2 * r2 + r3 + r4
        else:
            loss = lambda_class * crossentropy_loss + lambda_ae * re + lambda_1 * r2 + lambda_2 * re
        
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

def save_images(prototype_path, decoding_path, prototypes, subprototypes, decoding, epoch):
    if not os.path.exists(prototype_path):
        os.makedirs(prototype_path)
    if not os.path.exists(decoding_path):
        os.makedirs(decoding_path)
   
    save_image(prototypes, prototype_path+'seed{}prot{}.png'.format(args.seed, epoch), nrow=5, normalize=True)
    save_image(decoding, decoding_path+'seed{}dec{}.png'.format(args.seed, epoch), nrow=5, normalize=True)
    if subprototypes is not None: 
        save_image(subprototypes, prototype_path+'subprot{}.png'.format(epoch), nrow=3, normalize=True )

def train_MNIST(hierarchical=False, n_prototypes=15, n_sub_prototypes =15, 
                latent_size=40, n_classes=10,
                learning_rate=0.0001, training_epochs=1500, 
                batch_size=250, save_every=25, sigma=4, alpha=20):
    # Prepare file
    f = open("results_s" + str(args.seed ) + ".txt", "w")
    f.write(', '.join([str(x) for x in [hierarchical, n_prototypes, latent_size, learning_rate]]))
    f.write('\n')
    f.close()

    # Load data
    train_data = MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                            ]))
    test_data = MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                            ]))

    ### Initialize the model and the optimizer.
    if hierarchical:
        proto = HierarchyModel(n_prototypes, latent_size, n_classes, n_sub_prototypes)
    else:
        proto = PrototypeModel(n_prototypes, latent_size, n_classes)

    proto = proto.to(device)
    optim = torch.optim.Adam(proto.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_data, batch_size=batch_size)

    # Run for a number of epochs
    for epoch in range(training_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        it = 0
        
        #it, epoch_loss, epoch_acc, dec = run_epoch(hierarchical, sigma, alpha, proto, dataloader, optim, it, epoch_loss, epoch_acc)
        it, epoch_loss, epoch_acc, dec = run_epoch_n(sigma, alpha, proto, dataloader, optim, it, epoch_loss, epoch_acc)

        # To save time
        if epoch % save_every == 0:
        # Get prototypes and decode them to display
            prototypes = proto.prototype.get_prototypes()
            prototypes = prototypes.view(-1, 10, 2, 2)
            imgs = proto.decoder(prototypes)

            subprototypes = None
            if hierarchical:
                subprototypes = proto.prototype.get_sub_prototypes()
                for i in range(len(subprototypes)):
                    if i == 0:
                        subprotoset = subprototypes[i].view(-1,10,2,2)
                    else:
                        subprotoset = torch.cat([subprotoset, subprototypes[i].view(-1,10,2,2)])
                subprototypes = proto.decoder(subprotoset)

            # Save images
            save_images(prototype_path, decoding_path, imgs, subprototypes, dec, epoch)

            # Save model
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(proto, model_path+"proto.pth")

        # Print statement to check on progress
        with open("results_s" + str(args.seed ) + ".txt", "a") as f:
            text = "Epoch: " + str(epoch) + " loss: " + str(epoch_loss / it) + " acc: " + str(epoch_acc/it)
            print(text)
            f.write(text)
            f.write('\n')

    # Test data
    proto.eval()
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    test_accuracy = 0.0
    test_loss = 0.0
    it = 0
    for i, (images, labels) in enumerate(test_dataloader):
        it += 1
        images = images.to(device)
        labels = labels.to(device)
        oh_labels = one_hot(labels)

        # Forward pass
        if hierarchical:
            _, decoding, (r1, r2, r3, r4, c) = proto.forward(images)
        else:
            _, decoding, (r1, r2, c) = proto.forward(images)

        ce = nn.CrossEntropyLoss()
        # reconstruction error g(f(x)) and x
        subtr = (decoding - images).view(-1, 28*28)
        re = torch.mean(torch.norm(subtr, dim=1))

        crossentropy_loss = ce(c, torch.argmax(oh_labels, dim=1))
        if hierarchical:
            loss = lambda_class * crossentropy_loss + lambda_ae * re + \
                lambda_1 * r1 + lambda_2 * r2 + r3 + r4
        else:
            loss = lambda_class * crossentropy_loss + lambda_ae * re + lambda_1 * r2 + lambda_2 * re
        
        test_loss += loss.item()
        preds = torch.argmax(c,dim=1)
        corr = torch.sum(torch.eq(preds,labels))
        size = labels.shape[0]
        test_accuracy += corr.item()/size
    with open("results_s" + str(args.seed ) + ".txt", "a") as f:
        text = "Testdata loss: " +  str(test_loss/it) + " acc: " + str(test_accuracy/it)
        print(text)
        f.write(text)
        f.write('\n')
    

train_MNIST(hierarchical=False, batch_size=250)