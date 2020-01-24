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
from helper import check_path

# Global parameters for device and reproducibility
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproduction')
parser.add_argument('--dir', type=str, default='anna',
                        help='main directory to save intermediate results')
parser.add_argument("--hier", type=bool, nargs='?',const=True, default=False, help='Hierarchical turned on')                
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

check_path(args.dir)

model_path = args.dir + '/models/'
prototype_path = args.dir + '/prototypes/'
decoding_path = args.dir + '/decoding/'
results_path = args.dir +'/results/'

check_path(model_path)
check_path(prototype_path)
check_path(decoding_path)
check_path(results_path)

# Loss weights for cross entropy, reconstruction loss and the two extra terms as described in the paper
lambda_class_sup = 20 # CE for supers
lambda_class_sub = 20 # CE for subs
lambda_ae = 1
lambda_1 = 1
lambda_2 = 1
lambda_3 = 1
lambda_4 = 1

def run_epoch(hierarchical, sigma, alpha,                     # Model parameters
        model, dataloader, optimizer,                         # Training objects
        iteration, epoch_loss, epoch_accuracy, sub_accuracy): # Evaluation 

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
            _, decoding, (sub_c, sup_c, r1, r2, r3, r4) = model.forward(images)
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
        
        if hierarchical:
            sup_ce = ce(sup_c, torch.argmax(oh_labels, dim=1))
            # Extra cross entropy for second linear layer
            sub_ce = ce(sub_c, torch.argmax(oh_labels, dim=1))

            # Actual loss
            loss = lambda_class_sup * sup_ce + \
                lambda_ae * re + \
                lambda_class_sub * sub_ce + \
                lambda_1 * r1 + \
                lambda_2 * r2 + \
                lambda_3 * r3 + \
                lambda_4 * r4
        else:
            crossentropy_loss = ce(c, torch.argmax(oh_labels, dim=1))
            loss = lambda_class_sup * crossentropy_loss + \
            lambda_ae * re + \
            lambda_1 * r1 +  \
            lambda_2 * r2

        if(hierarchical):
            # For super prototype cross entropy term
            epoch_loss += loss.item()
            preds = torch.argmax(sup_c,dim=1)
            corr = torch.sum(torch.eq(preds,labels))
            size = labels.shape[0]
            epoch_accuracy += corr.item()/size

            # Also for sub prototype cross entropy term
            subpreds = torch.argmax(sub_c, dim=1)
            subcorr  = torch.sum(torch.eq(subpreds, labels))
            sub_accuracy += subcorr.item()/size
        else:
            # For prototype cross entropy term
            epoch_loss += loss.item()
            preds = torch.argmax(c,dim=1)
            corr = torch.sum(torch.eq(preds,labels))
            size = labels.shape[0]
            epoch_accuracy += corr.item()/size

        # Do backward pass and ADAM steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return iteration, epoch_loss, epoch_accuracy, sub_accuracy, decoding

def save_images(prototype_path, decoding_path, prototypes, subprototypes, decoding, epoch):
    save_image(prototypes, prototype_path+'prot{}.png'.format(epoch), nrow=5, normalize=True)
    if subprototypes is not None: 
        save_image(subprototypes, prototype_path+'subprot{}.png'.format(epoch), nrow=5, normalize=True )

def train_MNIST(hierarchical=False, n_prototypes=10, n_sub_prototypes = 30, 
                latent_size=40, n_classes=10,
                learning_rate=0.001, training_epochs=1500, 
                batch_size=250, save_every=1, sigma=4, alpha=20):
    # Prepare file
    f = open(results_path + "results_s" + str(args.seed ) + ".txt", "w")
    f.write(', '.join([str(x) for x in [hierarchical, n_prototypes, n_sub_prototypes, 
                    latent_size, learning_rate]]))
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
        sub_acc   = 0.0
        it = 0

        it, epoch_loss, epoch_acc, sub_acc, dec = run_epoch(hierarchical, sigma, alpha, proto, dataloader, 
                                                optim, it, epoch_loss, epoch_acc, sub_acc)

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
            text = "Epoch: " + str(epoch) + " loss: " + str(epoch_loss / it) + " acc: " + str(epoch_acc/it) + " sub_acc: " + str(sub_acc/it)
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
            _, decoding, (sub_c, sup_c, r1, r2, r3, r4) = model.forward(images)
        else:
            _, decoding, (r1, r2, c) = model.forward(images)

        ce = nn.CrossEntropyLoss()
        # reconstruction error g(f(x)) and x
        subtr = (decoding - images).view(-1, 28*28)
        re = torch.mean(torch.norm(subtr, dim=1))
        
        if hierarchical:
            sup_ce = ce(sup_c, torch.argmax(oh_labels, dim=1))
            # Extra cross entropy for second linear layer
            sub_ce = ce(sub_c, torch.argmax(oh_labels, dim=1))

            # Actual loss
            loss = lambda_class_sup * sup_ce + \
                lambda_ae * re + \
                lambda_class_sub * sub_ce + \
                lambda_1 * r1 + \
                lambda_2 * r2 + \
                lambda_3 * r3 + \
                lambda_4 * r4
        else:
            crossentropy_loss = ce(c, torch.argmax(oh_labels, dim=1))
            loss = lambda_class1 * crossentropy_loss + \
            lambda_ae * re + \
            lambda_1 * r1 +  \
            lambda_2 * r2

        if(hierarchical):
            # For super prototype cross entropy term
            test_loss += loss.item()
            preds = torch.argmax(sup_c,dim=1)
            corr = torch.sum(torch.eq(preds,labels))
            size = labels.shape[0]
            test_accuracy += corr.item()/size

            # Also for sub prototype cross entropy term
            subpreds = torch.argmax(sub_c, dim=1)
            subcorr  = torch.sum(torch.eq(subpreds, labels))
            sub_accuracy += subcorr.item()/size
        else:
            # For prototype cross entropy term
            test_loss += loss.item()
            preds = torch.argmax(c,dim=1)
            corr = torch.sum(torch.eq(preds,labels))
            size = labels.shape[0]
            test_accuracy += corr.item()/size

    with open(results_path + "results_s" + str(args.seed ) + ".txt", "a") as f:
        text = "Testdata loss: " +  str(test_loss/it) + " acc: " + str(test_accuracy/it)
        print(text)
        f.write(text)
        f.write('\n')
    

if (args.hier):
    train_MNIST(hierarchical=True, n_sub_prototypes=20)
else:
    train_MNIST()
