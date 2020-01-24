import torch
import os
"""
Code by Oscar Li
github.com/OscarcarLi/PrototypeDL
"""

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_of_distances(X, Y):
    '''
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    '''
    XX = torch.reshape(list_of_norms(X), shape=(-1, 1))
    YY = torch.reshape(list_of_norms(Y), shape=(1, -1))
    output = XX + YY - 2 * (X @ Y.t())

    return output

def list_of_norms(X):
    '''
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    '''
    return (torch.pow(X, 2)).sum(axis=1)

"""h = list_of_distances(
    torch.tensor([[1,1,1],[5,4,4],[1,2,3]]), 
    torch.tensor([[4,4,4],[3,3,3],[2,2,2],[1,1,1]]))
print(h)"""

