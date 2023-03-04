import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import plotly.graph_objects as go

import torch

@torch.no_grad()
def plot_decision_bounary(model, X, y, ax=None, figsize=(6, 6)):
    """Plot the decision boundary of a model."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the points
    plot_points(y, X, ax)

    # Plot the decision boundary
    x1 = torch.linspace(X[:,0].min(), X[:,0].max(), 100)
    x2 = torch.linspace(X[:,1].min(), X[:,1].max(), 100)
    X1, X2 = torch.meshgrid(x1, x2)
    X_grid = torch.stack([X1.ravel(), X2.ravel()], axis=1)

    # Add a column of ones to X_grid
    # X_grid = torch.cat([torch.ones(X_grid.shape[0], 1), X_grid], dim=1)

    # Apply model as a function to each row of X_grid
    y_grid = model(X_grid).detach().numpy()
    ax.contourf(X1, X2, y_grid.reshape(X1.shape), cmap=cm.coolwarm, alpha=0.4)

    ax.legend()
    return ax

def plot_points(y, X, ax=None, figsize=(6, 6)):
    """Plot the points in X with the corresponding labels in y."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    idx = y.flatten() == 1
    ax.scatter(X[idx,0], X[idx,1], label='1', alpha=.75, s=20)
    ax.scatter(X[~idx,0], X[~idx,1], label='0', alpha=.75, s=20) 

# Plot contours of a 2D function
def plot_contours(func, xrange, yrange, ax=None, levels=20, logscale=True, 
                  cmap=cm.coolwarm):
    """Plot the contours of a 2D function."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    x = np.linspace(*xrange)
    y = np.linspace(*yrange)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]
    Z = np.array([func(xy) for xy in XY])

    Z = Z.reshape(X.shape)
    my_ticker = ticker.LogLocator() if logscale else None
    if logscale:
        # Z = np.log(Z)
        levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), levels)
    cnt = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, locator=my_ticker, alpha=0.75)

    # cbar = plt.gcf().colorbar(cnt)
    # cbar.locator = ticker.LogLocator(10)
    # cbar.set_ticks(cbar.locator.tick_values(Z.min(), Z.max()))
    # cbar.minorticks_off()
    return ax, cnt


def our_gd(loss, beta, lr=0.1, n_steps=50):
    """Perform gradient descent on a loss function."""
    loss_hist = np.zeros(n_steps)
    beta_list = [beta.clone()]
    for i in range(n_steps):
        curr_loss = loss(beta)
        loss_hist[i] = curr_loss.item()

        curr_loss.backward()

        with torch.no_grad():   
            beta -= lr * beta.grad

        beta.grad.zero_()
        beta_list.append(beta.clone())
    return loss_hist, beta_list

def gd_with_momentum(loss, beta, momentum=0, lr=0.1, n_steps=50):
    """Perform gradient descent on a loss function."""
    loss_hist = np.zeros(n_steps)
    beta_list = [beta.clone()]
    z = torch.zeros_like(beta)
   
    for i in range(n_steps):
        curr_loss = loss(beta)
        loss_hist[i] = curr_loss.item()

        curr_loss.backward()

        with torch.no_grad():
            z = momentum * z + beta.grad
            beta -= lr * z 
            # beta -= lr * beta.grad # without momentum

        beta.grad.zero_()
        beta_list.append(beta.clone())
    return loss_hist, beta_list


def plot_gd_trajectory(theta_list, loss, xrange=(-5, 5), yrange=(-5, 5),
                       ax=None, figsize=(6, 6)):
    """Plot the trajectory of gradient descent."""
    # Concatenate theta_list into a tensor of shape (n_steps+1, 2) and make into a numpy array
    theta_tensor = torch.stack(theta_list).detach().numpy()

    # Plot a scatter plot of the rows of theta_tensor
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Rewrite with black color
    ax.scatter(theta_tensor[: , 0], theta_tensor[: , 1], s=20, c='k')    
    
    # Add line scatter plot of the rows of theta_tensor
    ax.plot(theta_tensor[: , 0], theta_tensor[: , 1], c='k')

    # Plot the countours of the loss function
    plot_contours(loss, xrange, yrange, ax, levels=20)
    
    # ax.set_xlabel(r'$\theta_1$')
    # ax.set_ylabel(r'$\theta_2$')
    return ax

def plot_gd_3d_trajectory(theta_tensor):   
    #fig = go.Figure()
    fig = go.Figure(data=[go.Scatter3d(x=theta_tensor[: , 0], y=theta_tensor[: , 1], z=theta_tensor[: , 2], mode='lines')])
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=10,
            t=10,
            pad=4
        ),
    )
    # add scatter plot to figure
    #fig.add_trace(go.Scatter3d(x=theta_tensor[: , 0], y=theta_tensor[: , 1], z=theta_tensor[: , 2], mode='lines'))

    fig.show()



# def train(model, train_loader, optimizer, criterion, device):
#     """Train a model on a training set."""
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     train_loss /= len(train_loader.dataset)
#     return train_loss