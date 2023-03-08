import torch
import torch.nn as nn

# implement the Nadaraya-Watson kernel regression in PyTorch
class NW(nn.Module):
    def __init__(self, X_train, y_train, h=1, kernel=None):
        super().__init__()
        self.h = h
        self.X_train = X_train
        self.y_train = y_train.type_as(self.X_train)
        if kernel is None:
            kernel = lambda x: torch.exp(-x**2 / (2 * h**2))
        self.kernel = kernel

    def forward(self, X_test):
        X_test = X_test.type_as(self.X_train)
        dist = torch.cdist(X_test, self.X_train)
        weights = self.kernel(dist)
        # return torch.sum(weights * self.y_train, dim=1) / torch.sum(weights, dim=1) 
        return (weights @ self.y_train) / (weights @ torch.ones_like(self.y_train))
    
# A dense layer
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, activation=torch.tanh):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation
        
    def forward(self, x):
        x = self.fc(x)
        return self.activation(x)

