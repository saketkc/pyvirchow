from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.color import rgb2gray


def plot_input_mask_grid(X, Y):
    """Plot input patches with their patches

    Parameters
    ----------
    X: list
       List of Input RGB images (256x256x3)

    Y: list
       List of Input binar mask (256x256)
       1 = tumor (white)
       0 = normal (black)

    """
    x_size = X.shape[0]
    f, axes = plt.subplots(int(np.ceil(x_size) / 4), 8, figsize=(20, 20))
    ax = axes.flatten()
    for i in range(0, 2 * x_size - 1):
        ax[i].imshow(X[i])
        ax[i + 1].imshow(Y[i])
        ax[i].axis('off')
        ax[i + 1].axis('off')
    f.set_title('NoTruth Masks 32x256x256x1')
    return f, axes


def plot_blend(patch, prediction, ax, alpha=0.75):
    """Plot patch blended with tumor prediction probabilities

    Parameters
    ----------
    patch: array_like
           RGB WSI patch (256x256x3)
    prediction: array_like
                Probability prediction values for each pixel
    ax: matplotlib.Axes
    alpha: float
           Mixing parameter
    """

    dx, dy = 0.05, 0.05
    x = np.arange(0, patch.shape[1] - 1, dx)
    y = np.arange(0, patch.shape[0] - 1, dy)
    xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
    extent = xmin, xmax, ymin, ymax

    Z1 = rgb2gray(patch)
    Z2 = prediction

    im1 = ax.imshow(Z1, cmap='gray', extent=extent)
    im2 = ax.imshow(
        Z2, cmap='coolwarm', alpha=alpha, vmin=0.0, vmax=1.0, extent=extent)

    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label(
        'Probability of pixel being a tumor', weight='bold', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    ax.axis('off')
    return ax


def plot_patch_with_pred(patch,
                         truth=None,
                         prediction=None,
                         title_str='',
                         alpha=0.6):
    """

    Parameters
    ----------
    patch: array_like
           RGB image
    truth: array_like
           Binary mask
    prediction: array_like
                256x256x1, per-pixel tumor probability
    """
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, width_ratios=[10, 10, 19, 1])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax4 = plt.subplot(gs[:, 2])
    axc = plt.subplot(gs[:, 3])

    ax0.imshow(patch)
    ax0.set_title('Original')

    if truth:
        ax1.imshow(truth.argmax(axis=2), cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Truth mask (white=tumor, black=normal)')

    p = ax2.imshow(prediction, cmap='coolwarm', vmin=0, vmax=1)
    ax2.set_title('Prediction heatmap')

    ax3.imshow((prediction > 0.5).astype(np.int), cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Prediction mask (white=tumor, black=normal)')

    plot_blend(patch, prediction, ax4, alpha)
    ax4.set_title('Original+Prediction blend')

    fig.suptitle(title_str)
    fig.colorbar(p, cax=axc, orientation='vertical')
    axc.set_title('Probability pixel is tumor')
