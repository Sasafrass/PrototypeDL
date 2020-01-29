"""
Code of function list_of_distances by Oscar Li
github.com/OscarcarLi/PrototypeDL
"""
import os
import torch

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_of_distances(x, y):
    """
    Given a list of vectors, x = [x_1, ..., x_n], and another list of vectors,
    y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    """
    x_reshape = torch.reshape(list_of_norms(x), shape=(-1, 1))
    y_reshape = torch.reshape(list_of_norms(y), shape=(1, -1))
    output = x_reshape + y_reshape - 2 * (x @ y.t())

    return output

def list_of_norms(x):
    """
    x is a list of vectors x = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    """
    return (torch.pow(x, 2)).sum(axis=1)
