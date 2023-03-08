# import MNIST data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# def load_mnist(batch_size=128, valid_size=0.2, shuffle=True, random_seed=42):
#     """Load the MNIST dataset and split it into training and validation sets."""
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.5,), (0.5,)),
#                                     ])

#     trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
#     testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)

#     # obtain training indices that will be used for validation
#     num_train = len(trainset)
#     indices = list(range(num_train))
#     split = int(np.floor(valid_size * num_train))
#     if shuffle:
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)
#     train_idx, valid_idx = indices[split:], indices[:split]

#     # define samplers for obtaining training and validation batches
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)

#     # prepare data loaders (combine dataset and sampler)
#     trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
#     validloader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

#     return trainloader, validloader, testloader


def load_mnist(batch_size=128, shuffle=True):
    """Load the MNIST dataset"""

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(.5, .5),
                                    ])

    # Download and load the training data
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=shuffle)

    # Download and load the test data
    testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=shuffle)

    return trainloader, testloader

def load_breast_cancer(data_path = None):
    if data_path is None:
        # data_path = '../data/'
        data_path = Path(__file__).parent.parent / 'data/'

    df = pd.read_csv(data_path / 'breast_cancer.csv')
    train_idx = pd.read_csv(data_path / 'breast_cancer_train_indices.csv')['train_idx'].to_numpy()

    # Select features 
    df = df[['radius_mean', 'texture_mean', 'diagnosis']]
    
    # Drop rows with missing values
    df = df.dropna()

    X = df[['radius_mean', 'texture_mean']].to_numpy()
    y = df['diagnosis'].to_numpy()

    # Convert the target to 0 and 1
    y = (y == 'M').astype(int)

    # Split the data into training and test sets
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = np.delete(X, train_idx, axis=0), np.delete(y, train_idx, axis=0)

    return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)
