from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..io.operations import read_as_rgb
import numpy as np
from numpy import linalg as LA
import six
import spams


def RGB2OD(image):
    """Convert Intensities to Optical Density"""
    assert np.issubdtype(image.dtype, np.uint8)
    image[np.where(image == 0)] = 1
    return (-np.log(image / 255.0))


def OD2RGB(OD):
    """Convert optical density back to RGB"""
    return 255 * np.exp(-OD)


class MacenkoNormalization(object):
    def __init__(self, beta=0.15, alpha=1, lambda1=0.01):
        """Implementation of Macenko's method.
        See: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf

        Input: RGB Slide
        1. Convert RGB to OD
        2. Remove data with OD intensity less than beta
        3. Calculate SVD on the OD tuples
        4. Create plane from the SVD directions corresponding to the
        two largest singular values
        5. Project data onto the plane, and normalize to unit length
        6. Calculate angle of each point wrt the first SVD direction
        7. Find robust extremes (alpha th and (100 - alpha)th
        percentiles) of the  angle
        8. Convert extreme values back to OD space


        More formally, consider I is a n x 3 matrix (n rows, 3 columns for RGB),
        we look for two matrixes C and S such that:
            C = n x 2 : Matrix of Concentrations in Channel H[1] and Channel E[2]
            S = 2 x 3 : Matrix of Stain vectors, 3 channels for H and 3 channels for E

        This is achieved by doing a non negative matrix factorization of I

        min_{C,S} ||I-CS||_2^2

        """
        self.beta = beta
        self.alpha = alpha
        self.lambda1 = lambda1
        self.OD = None

    def fit(self, target_image):
        """Fit attributes to target image.

        Parameters
        ----------
        target_image:

        """
        if isinstance(target_image, six.string_types):
            target_image = read_as_rgb(target_image)
        self.h, self.w, self.c = target_image.shape
        self.target_stain_matrix = self.get_stain_matrix(target_image)
        self.target_concentrations = self.get_concentrations(
            target_image, self.target_stain_matrix)

    def get_stain_matrix(self, source_image):
        """ OD = SV. Get V

        Parameters
        ----------
        source_image: array_like
                      np.unit8 array of rgb values

        Returns
        -------

        stain_matrix: array_like
                      2 x M matrix for a N x M matrix

        """
        OD = RGB2OD(source_image)
        OD = OD.reshape((-1, 3))
        OD = (OD[(OD > self.beta).any(axis=1), :])
        self.OD = OD
        # do-PCA
        OD_cov = np.cov(OD, rowvar=False)
        w, v = LA.eigh(OD_cov)
        # Project OD into first two directions
        v_first_two = v[:, [1, 2]]
        projected = np.dot(OD, v_first_two)
        # find the min and max vectors and project back to OD space
        angle = np.arctan2(projected[:, 1], projected[:, 0])

        min_angle = np.percentile(angle, self.alpha)
        max_angle = np.percentile(angle, 100 - self.alpha)

        Vmin = np.dot(v_first_two,
                      np.array([np.cos(min_angle),
                                np.sin(min_angle)]))
        Vmax = np.dot(v_first_two,
                      np.array([np.cos(max_angle),
                                np.sin(max_angle)]))

        # For the sake of consistency,
        # we assume that the first vector with more concentation is the
        # Hematoxylin vector. The motivation for this is our visual understanding
        # that the purple Hematoxylin generally appears to be dominant over
        # the pink Eosin.
        if Vmax[0] > Vmin[0]:
            HE = np.array([Vmax, Vmin])
        else:
            HE = np.array([Vmin, Vmax])
        HE = HE / np.linalg.norm(HE, axis=1)[:, None]
        return HE

    def get_concentrations(self, image, stain_matrix):
        """Get concentration matrix.

        Parameters
        ----------
        image: array_like
               rgb

        Returns
        -------

        concentration: array_like
                       N x 2 matrix for an N x M case.
        """
        OD = RGB2OD(image).reshape((-1, 3))
        coefs = spams.lasso(
            OD.T, D=stain_matrix.T, mode=2, lambda1=self.lambda1,
            pos=True).toarray().T
        return coefs

    def transform(self, source_images):
        """Transform source image to target.

        Parameters
        ----------
        source_image: list(array_like)
                      np.unit8 rgb input

        Returns
        -------
        reconstructed: array_like
                       np.uint8 transformed image

        """
        if not isinstance(source_images, list):
            source_images = [source_images]
        normalized_images = []
        for source_image in source_images:
            source_stain_matrix = self.get_stain_matrix(source_image)
            source_concentrations = self.get_concentrations(
                source_image, source_stain_matrix)

            maxC_source = np.percentile(
                source_concentrations, 99, axis=0).reshape((1, 2))
            maxC_target = np.percentile(
                self.target_concentrations, 99, axis=0).reshape((1, 2))
            source_concentrations *= (maxC_target / maxC_source)
            reconstructed = np.dot(source_concentrations,
                                   self.target_stain_matrix).reshape(
                                       source_image.shape)

            reconstructed = OD2RGB(reconstructed).reshape(
                source_image.shape).astype(np.uint8)
            normalized_images.append(reconstructed)
        return normalized_images

    def get_hematoxylin_channel(self, source_image=None, total=True):
        """Get hematoxylin channel concentration of source or target image.

        If source image is providedd, then the returned value
        represents the target stain vector remains the same,
        the source image concentration is then calculated by an inverse calculation C = IS^{-1}


        If no source image is provided, it simply returns the concentration
        channel of the target image, first column in this case.

        """
        if source_image is not None:
            h, w, _ = source_image.shape
            stain_matrix_source = self.get_stain_matrix(source_image)
            concentrations = self.get_concentrations(source_image,
                                                     stain_matrix_source)
        else:

            h, w, _ = self.h, self.w, self.c
            concentrations = self.target_concentrations
        if total:
            H = np.multiply(
                np.array(concentrations[:, 0], ndmin=2).T,
                np.reshape(self.target_stain_matrix[0, :], (1, 3))).reshape(
                    h, w, 3)
        else:
            H = concentrations[:, 0].reshape(h, w)
        H = OD2RGB(H)
        return H

    def get_eosin_channel(self, source_image=None, total=True):
        """Get eosin channel concentration of source or target image.

        If source image is providedd, then the returned value
        represents the target stain vector remains the same,
        the source image concentration is then calculated by an inverse calculation C = IS^{-1}

        If no source image is provided, it simply returns the concentration
        channel of the target image, second column in this case.

        """
        if source_image is not None:
            h, w, _ = source_image.shape
            stain_matrix_source = self.get_stain_matrix(source_image)
            concentrations = self.get_concentrations(source_image,
                                                     stain_matrix_source)
        else:
            h, w, _ = self.h, self.w, self.c
            concentrations = self.target_concentrations
        if total:
            E = np.multiply(
                np.array(concentrations[:, 1], ndmin=2).T,
                np.reshape(self.target_stain_matrix[1, :], (1, 3))).reshape(
                    h, w, 3)
        else:
            E = concentrations[:, 1].reshape(h, w)
        E = OD2RGB(E)
        return E

    def get_hematoxylin_stain(self, vis=True):
        """ Get the 3 channel values belonging to the hematoxylin stain

        Parameters
        ----------
        vis: bool
             Should visualize?

        Returns
        -------
        H: array_like
           A length 3 vector of RGB values
        """
        H = self.target_stain_matrix[0, :]
        H = OD2RGB(H)
        if vis:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_axis_off()
            ax.plot([0, 1], [1, 1], c=H / 255, linewidth=40)
        return H

    def get_eosin_stain(self, vis=True):
        """ Get the 3 channel values belonging to the eosin stain

        Parameters
        ----------
        vis: bool
             Should visualize?

        Returns
        -------
        E: array_like
           A length 3 vector of RGB values
        """
        E = self.target_stain_matrix[1, :]
        E = OD2RGB(E)
        if vis:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_axis_off()
            ax.plot([0, 1], [1, 1], c=E / 255, linewidth=40)
        return E
