from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
from ..io import WSIReader
from . import otsu_thresholding, plot_contours


class TissuePatch(object):
    def __init__(self, reference_image):
        if isinstance(reference_image, six.string_types):
            self.ref_img = WSIReader(reference_image)
        else:
            assert isinstance(
                reference_image,
                WSIReader), 'input image should be string or WSIReader'
            self.ref_img = reference_image
        self.otsu_thresholed = self.threshold()
        self.ref_level = 9

    def threshold(self, level=9, channel='saturation'):
        """Perform otsu thresholding

        Parameters
        ----------
        level: int
               What level of the pyramid are we going to threshold

        channel: string
                 'hue'/'saturation'/'value'
        """
        self.ref_level = level
        self.ref_magnification = self.ref_img.magnifications[self.ref_level]
        zoomed_out_patch_rgb = self.ref_img.get_patch_by_level(
            0, 0, level, None)
        self.otsu_thresholded = otsu_thresholding(zoomed_out_patch_rgb)
        return self.otsu_thresholded

    def draw_contours(self):
        """Draw contours and rectangular boxes"""
        plot_contours(self.otsu_thresholded)

    def extract_masked_patch(self,
                             x0,
                             y0,
                             target_level=None,
                             target_magnification=None,
                             patch_size=299):
        """Extract patch at a particular level

        Parameters
        ----------
        x0: int
            x coordinate of top left of the patch to be extracted

        y0: int
            y coordinate of top left of the patch to be extracted

        target_level: int
                      At what level[0-9] should the patch be extracted

        target_magnification: int
                      At what magnification should the patch be extracted

        Either of target_level or target_magnification should be specified

        """
        if target_level is None and target_magnification is None:
            raise ValueError(
                'At least one of target_level and target_magnification\
                should be specified.')

        if target_level:
            # Prefer this if it is specified
            target_magnification = self.ref_img.magnifications[target_level]
        magnification_factor = target_magnification / self.ref_img.level0_mag
        x0 = x0 * magnification_factor
        y0 = y0 * magnification_factor
        final_size = magnification_factor * patch_size
        patch = self.otsu_thresholded[x0:x0 + final_size, y0:y0 + final_size]
        return patch
