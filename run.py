import argparse
from train import train_MNIST, load_and_test

# Global parameters for device and reproducibility
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproduction')
parser.add_argument('--dir', type=str, default='my_own_model',
                        help='main directory to save intermediate results')
parser.add_argument("--hier", type=bool, nargs='?',const=True, default=False, help='Hierarchical mode turned on')                
args = parser.parse_args()


if (args.hier):
    train_MNIST(hierarchical=True, n_sub_prototypes=20, directory=args.dir, seed = args.seed, training_epochs=900, underrepresented_class=9)
else:
    train_MNIST(n_prototypes=15, directory=args.dir, seed = args.seed, training_epochs=900, underrepresented_class=9)