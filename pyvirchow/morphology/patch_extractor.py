import os
import six
from .operations import otsu_thresholding, plot_contours, contours_and_bounding_boxes
import numpy as np
import pickle
from ..io.operations import WSIReader
import matplotlib.pyplot as plt


class TissuePatch(object):
    def __init__(self, reference_image, level=5, level0_mag=None):
        if isinstance(reference_image, six.string_types):
            self.ref_img = WSIReader(reference_image, level0_mag)
        else:
            assert (
                isinstance(reference_image, WSIReader) is True
            ), "input image should be string or WSIReader"
            self.ref_img = reference_image
        self.otsu_thresholded = self.threshold(level=level)

    def threshold(
        self,
        level=5,
        channel="saturation",
        open_kernel_size=5,
        close_kernel_size=10,
        use_disk=True,
    ):
        """Perform otsu thresholding

        Parameters
        ----------
        level: int
               What level of the pyramid are we going to threshold

        channel: string
                 'hue'/'saturation'/'value'
        """
        self.ref_thresholding_level = level
        self.ref_magnification = self.ref_img.magnifications[
            self.ref_thresholding_level
        ]
        self.zoomed_out_patch_rgb = self.ref_img.get_patch_by_level(0, 0, level, None)
        self.magnification_factor = self.ref_magnification / self.ref_img.level0_mag
        self.otsu_thresholded = otsu_thresholding(
            self.zoomed_out_patch_rgb,
            channel,
            open_kernel_size,
            close_kernel_size,
            use_disk,
        )
        return self.otsu_thresholded

    def get_bounding_boxes(self):
        """Get bounding boxes for the reference zoomed out version."""
        _, bounding_boxes = contours_and_bounding_boxes(
            self.otsu_thresholded, self.zoomed_out_patch_rgb
        )
        return bounding_boxes

    def draw_contours(self):
        """Draw contours and rectangular boxes"""
        ax, self.tissue_bounding_boxes = plot_contours(
            self.otsu_thresholded, self.zoomed_out_patch_rgb
        )

    def save_mask(self, savedir):
        """Save tissue patch.

        savedir: string
                Path to directory where to save the pickle
        """
        ID = self.ref_img.uid
        os.makedirs(savedir, exist_ok=True)
        filepath = os.path.join(savedir, ID + "_TissuePatch.npy")
        np.save(filepath, np.array(self.otsu_thresholded))
        pickler = open(filepath.replace(".npy", ".pickle"), "wb")
        pickle.dump(self, pickler)
        pickler.close()

    def load(self, path):
        """Load tissue patch from pickled file.

        Parameters
        ----------
        path: string
              Path to pickle file

        Returns
        -------
        model: class TissuePatch
               loaded object
        """
        pickler = pickle.load(open(path, "rb"))
        self.ref_thresholding_level = pickler.ref_thresholding_level
        self.ref_magnification = pickler.ref_magnification
        self.otsu_thresholded = pickler.otsu_thresholded
        self.ref_img = WSIReader(pickler.filepath, self.ref_magnification)

    def visualize_mask(self, figsize=(10, 10)):
        """Visualize masked out region"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.imshow(self.otsu_thresholded, cmap="gray")
        return ax

    def __getstate__(self):
        """Return state values to be pickled."""
        uid = self.ref_img.uid
        return (
            self.ref_thresholding_level,
            self.ref_magnification,
            self.zoomed_out_patch_rgb,
            self.magnification_factor,
            self.otsu_thresholded,
            uid,
            self.ref_img.filepath,
        )
