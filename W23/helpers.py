import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

import torch

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
