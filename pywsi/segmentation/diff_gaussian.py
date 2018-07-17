from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import numpy as np


def downsample(image, factor=2):
    return image[::factor, ::factor]


def laplace_of_gaussian(input_image,
                        foreground_mask,
                        sigma_min,
                        sigma_max,
                        n_octave_levels=3):

    eps = np.finfo('float').eps

    input_image = input_image.astype('float')
    image_dist_transform = distance_transform_edt(foreground_mask)

    sigma_upper_bound = 2 * image_dist_transform
    sigma_upper_bound = np.clip(sigma_upper_bound, sigma_min, sigma_max)

    delta_sigma = 2**(1.0 / n_octave_levels)
    sigma_ratio = sigma_max / sigma_min
    n_levels = np.ceil(np.log(sigma_ratio) / np.log(delta_sigma)).astype(int)

    sigma_prev = sigma_min
    convolution_prev = gaussian_filter(input_image, sigma_prev)
    sigma_upper_bound_cur = sigma_upper_bound.copy()

    dog_max = np.zeros_like(input_image)
    dog_max[:, :] = eps
    dog_octave_max = dog_max.copy()

    sigma_max = np.zeros_like(input_image)
    sigma_octave_max = np.zeros_like(input_image)

    n_level = 0
    n_octave = 0

    for level_cur in range(n_levels + 1):
        sigma_cur = sigma_prev * delta_sigma
        sigma_conv = np.sqrt(sigma_cur**2 - sigma_prev**2)
        sigma_conv /= 2**n_octave

        convolution_cur = gaussian_filter(convolution_prev, sigma_conv)
        dog_cur = convolution_cur - convolution_prev
        dog_cur[sigma_upper_bound_cur < sigma_prev] = eps
        pixels_to_update = np.where(dog_cur > dog_octave_max)
        if len(pixels_to_update[0]) > 0:
            dog_octave_max[pixels_to_update] = dog_cur[pixels_to_update]
            sigma_octave_max[pixels_to_update] = sigma_prev

        sigma_prev = sigma_cur
        convolution_prev = convolution_cur
        n_level += 1

        # Do additional processing at the end of each octave
        if level_cur == n_levels or n_level == n_octave_levels:
            # update maxima
            if n_octave_levels > 0:

                dog_octave_max_resized = resize(
                    dog_octave_max, dog_max.shape, order=0)

            else:

                dog_octave_max_resized = dog_octave_max

            max_pixels = np.where(dog_octave_max_resized > dog_max)
            if len(max_pixels[0]) > 0:

                dog_max[max_pixels] = \
                    dog_octave_max_resized[max_pixels]

                if n_octave_levels > 0:

                    sigma_octave_max_resized = resize(
                        sigma_octave_max, dog_max.shape, order=0)

                else:

                    sigma_octave_max_resized = sigma_octave_max

                sigma_max[max_pixels] = \
                    sigma_octave_max_resized[max_pixels]

            if n_level == n_octave_levels:

                convolution_prev = downsample(convolution_cur)
                sigma_upper_bound_cur = downsample(sigma_upper_bound_cur)

                dog_octave_max = downsample(dog_octave_max)
                sigma_octave_max = downsample(sigma_octave_max)

                n_level = 0
                n_octave += 1

    # set min vals to min response
    dog_max[dog_max == eps] = 0

    return dog_max, sigma_max
