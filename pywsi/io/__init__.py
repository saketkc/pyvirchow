import ntpath
import warnings
import cv2
import openslide
from openslide import OpenSlide


def path_leaf(path):
    """Get base and tail of filepath

    Parameters
    ----------
    path: string
          Path to split

    Returns
    -------
    tail: string
          Get tail of path
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_as_rgb(image_path):
    """Read image as RGB

    Parameters
    ----------
    image_path: str
                Path to image

    Returns
    -------
    rgb_image: array_like
               np.uint8 array of RGB values
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class WSIReader(OpenSlide):
    def __init__(self, image_path, level0_mag):
        """Read image as Openslide object

        Parameters
        ----------
        image_path: str
                    Path to image file

        level0_mag: int
                    Magnification at level 0 (most detailed)
        """
        super(WSIReader, self).__init__(image_path)
        inferred_level0_mag = self.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
        if inferred_level0_mag:
            if inferred_level0_mag != level0_mag:
                warnings.warn(
                    '''Inferred and provided level0 mag mismatch.
                              Provided {}, but found {}. Will use the latter.'''
                    .format(inferred_level0_mag, level0_mag), UserWarning)
                self.level0_mag = level0_mag
        else:
            self.level0_mag = level0_mag

        self.uid = path_leaf(image_path)
        width, height = self.dimensions
        self.width = width
        self.height = height
        self.magnifications = [
            self.level0_mag / downsample
            for downsample in self.level_downsamples
        ]

    def get_patch_by_level(self, xstart, ystart, level, patch_size=None):
        """Get patch by specifying magnification

        Parameters
        ----------
        xstart: int
                Top left pixel x coordinate
        ystart: int
                Top left pixel y coordinate
        magnification: int
                       Magnification to extract at
        patch_size: tuple
                    Patch size for renaming

        """
        if not patch_size:
            width, height = self.level_dimensions[level]
        else:
            width, height = patch_size
        patch = self.read_region((xstart, ystart), level,
                                 (width, height)).convert('RGB')
        return patch

    def get_patch_by_magnification(self,
                                   xstart,
                                   ystart,
                                   magnification,
                                   patch_size=None):
        """Get patch by specifying magnification

        Parameters
        ----------
        xstart: int
                Top left pixel x coordinate
        ystart: int
                Top left pixel y coordinate
        magnification: int
                       Magnification to extract at
        patch_size: tuple
                    Patch size for renaming

        """
        filtered_mag = list(
            filter(lambda x: x >= magnification, self.magnifications))
        # What is the possible magnification available?
        possible_mag = min(filtered_mag)
        possible_level = self.magnifications.index(possible_mag)
        # Rescale the size of image to match the new magnification
        if patch_size:
            rescaled_size = possible_mag / magnification * patch_size
            rescaled_width, rescaled_height = rescaled_size, rescaled_size
        else:
            rescaled_width, rescaled_height = self.level_dimensions[
                possible_level]
        print(possible_mag, possible_level)
        patch = self.read_region(
            (xstart, ystart), possible_level,
            (rescaled_width, rescaled_height)).convert('RGB')
        #patch = None
        return patch

    def show_all_properties(self):
        """Print all properties.
        """
        print('Properties')
        for key in self.properties.keys():
            print('{} : {}'.format(key, self.properties[key]))
