from scipy.stats import poisson
from skimage.color import rgb2gray
from sklearn import mixture
import numpy as np


def poisson_deconvolve(h_channel_image):
    """Separate baclground and foreground intensities
    by doing a poisson partition.

    For the i_th pixel h(i) = P0*P(i|0) + P1*P(i|1)
    where P0,P1 are prior probabilities of background and foreground.

    For a threshold t, the parameters are given by:
    P(t) = \sum_{i=0}^t h(i)
    \mu_0(t) = 1/P0(t) * \sum_{i=0}^t i*h(i)

    P1(t) = \sum_{i=t+1}^I_max h(i)
    \mu_1(t) = 1/P1(t) * \sum_{i=t+1}^I_max i*h(i)


    Parameters
    ----------
    h_channel_image: array_like
                     uint8 image (single channel)
    """
    assert np.issubdtype(h_channel_image.dtype, np.integer), "Input should be int dtype"
    eps = np.finfo("float").eps
    h, bin_edges = np.histogram(np.ravel(h_channel_image), bins=256, range=(0, 256))
    h_normalized = h.astype("float") / h.sum()

    cumul_sum = np.cumsum(h_normalized)
    product = np.multiply(bin_edges[:-1], h_normalized)
    cumul_product = np.cumsum(product)

    p0 = cumul_sum
    p0[p0 <= 0] = eps

    p1 = 1 - cumul_sum
    p1[p1 <= 0] = eps

    mu = np.mean(np.ravel(h_channel_image))

    mu0 = np.divide(cumul_product, p0) + eps
    mu1 = np.divide(cumul_product[-1] - cumul_product, p1) + eps

    cost = (
        mu
        - np.multiply(p0, np.log(p0) + np.multiply(mu0, np.log(mu0)))
        - np.multiply(p1, np.log(p1) + np.multiply(mu1, np.log(mu1)))
    )

    min_cost_index = np.argmin(cost)

    mu0_opt = mu0[min_cost_index]
    mu1_opt = mu1[min_cost_index]

    t_opt = bin_edges[min_cost_index]

    foreground = poisson.pmf(np.arange(0, 256), mu1_opt)[h_channel_image]
    background = poisson.pmf(np.arange(0, 256), mu0_opt)[h_channel_image]

    opts = {"t": t_opt, "mu0_opt": mu0_opt, "mu1_opt": mu1_opt}

    return (
        foreground.reshape(h_channel_image.shape),
        background.reshape(h_channel_image.shape),
        opts,
    )


def gmm_thresholding(image_rgb):
    """Perform thresholding based on Gaussian mixture models.

    The image is assumed to be a mixture of two gaussians.
    A lower mean sample belongs to the blobs while the higher mean
    shows the white background.

    Parameters
    ----------
    image_rgb: array_like
               RGB input

    Returns
    ------
    gmm_threshold: float
                   GMM mean (minimum) of the two mixing populations
    clf: sklearn.GaussianMixture
         The entire sklearn model
    """
    clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
    clf.fit(rgb2gray(image_rgb).flatten().reshape(-1, 1))
    return clf.means_.min(), clf
