from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
import six
from ..morphology.mask import get_common_interior_polygons
from .operations import get_annotation_polygons
from .operations import poly2mask
from .operations import translate_and_scale_polygon
from .operations import WSIReader

from skimage.filters import threshold_otsu
from openslide.deepzoom import DeepZoomGenerator

from shapely.geometry import Point as shapelyPoint
from shapely.geometry import Polygon as shapelyPolygon


def mask_to_categorical(y, num_classes=2, patch_size=256):
    """Convert binary mask to categorical array for Keras/TF.

    Parameters
    ----------
    y: array
       Input array
    num_classes: int
                 Number of classes (2)
    patch_size: int
                Original patch size

    Output
    ------
    y_cat: array
           Output array
    """
    from keras.utils.np_utils import to_categorical
    y_cat = to_categorical(
        y, num_classes=num_classes).reshape(y.shape[0], patch_size, patch_size,
                                            num_classes)
    return y_cat


def get_approx_tumor_mask(polygons,
                          thumbnail_nrow,
                          thumbnail_ncol,
                          patch_size=256):
    """Indicate whether a particular tile overlaps with tumor annotation.

    The method has an approx in its name, as the entire tile
    might not be coming from a tumor annotated region as parts of
    it might actually be normal. We just report
    here if the patch overlaps with a tumor annotation. It might have
    an overlap with a normal region as well, but that is filtered
    in a later method so we don't worry about it here.

    Parameters
    ----------
    polygons: dict
              dict with keys ['normal', 'tumor']
              as obtained from `get_annotation_polygons`
    thumbnail_nrow: int
                    Number of rows in the thumbnail image
    thumbnail_ncol: int
                    Number of columns in the thumbnail

    Returns
    -------
    mask: array
          tumor mask
    """
    scaled_tumor_polygons = []
    for tpol in polygons['tumor']:
        scaled = translate_and_scale_polygon(tpol, 0, 0, 1 / 256)
        scaled_tumor_polygons.append(scaled)
    polymasked = poly2mask(scaled_tumor_polygons,
                           (thumbnail_nrow, thumbnail_ncol))
    # Is any of the masked out points inside a normal annotated region?
    poly_x, poly_y = np.where(polymasked > 0)
    set_to_zero = []
    for px, py in zip(poly_x, poly_y):
        point = shapelyPoint(px, py)
        for npol in polygons['normal']:
            scaled = translate_and_scale_polygon(npol, 0, 0, 1 / 256)
            pol = shapelyPolygon(scaled.get_xy())
            if pol.contains(point):
                set_to_zero.append((px, py))

    if len(set_to_zero):
        set_to_zero = np.array(set_to_zero)
        polymasked[set_to_zero] = 0
    return polymasked


def create_tumor_mask_from_tile(tile_x, tile_y, polygons, patch_size=256):
    """Create a patch_size x patch_size mask from tile_x,y coordinates
    Parameters
    ----------
    tile_x, tile_y:  int
    polygons: dict
              ['normal', 'tumor'] with corresponding polygons

    Returns
    -------
    mask: array
          patch_size x patch_size binary mask

    """

    # Initiate a zero mask
    mask = np.zeros((patch_size, patch_size))
    #patch_polygon = shapelyRectangle(tile_x, tile_y, patch_size, patch_size)
    x_min = tile_x
    y_min = tile_y
    x_max = x_min + 256
    y_max = y_min + 256
    patch_polygon = shapelyPolygon([(x_min, y_min), (x_max, y_min),
                                    (x_max, y_max), (x_min, y_max)])

    # Is it overlapping any of the tumor polygons?
    is_inside_tumor = [
        patch_polygon.intersection(polygon.buffer(0))
        for polygon in polygons['tumor']
    ]

    # the patch will always be inside just one annotated boundary
    # which are assumed to be non-overlapping and hence we can just fetch
    # the first sample
    tumor_poly_index = None
    tumor_poly_coords = None
    for index, sample_intersection in enumerate(is_inside_tumor):
        if sample_intersection.area > 0:
            tumor_poly_index = index
            if sample_intersection.geom_type == 'Polygon':
                tumor_poly_coords = np.array(
                    sample_intersection.boundary.coords)
            elif sample_intersection.geom_type == 'MultiPolygon':
                tumor_poly_coords = []
                for p in sample_intersection:
                    tumor_poly_coords += p.boundary.coords
                tumor_poly_coords = np.array(tumor_poly_coords)
            elif sample_intersection.geom_type == 'GeometryCollection':
                tumor_poly_coords = []
                for p in sample_intersection:
                    if p.geom_type == 'LineString':
                        tumor_poly_coords += p.coords
                    elif p.geom_type == 'Polygon':
                        tumor_poly_coords += p.boundary.coords
                    else:
                        print('Found geom_type:{}'.format(p.geom_type))
                        raise ValueError('')
            else:
                print('Found geom_type:{}'.format(
                    sample_intersection.geom_type))
                raise ValueError('')
            break

    if tumor_poly_index is None:
        # No overlap with tumor so must return as is
        return mask

    # This path belongs to a tumor patch so set everything to one
    # Set these coordinates to one

    # Shift the tumor coordinates to tile_x, tile_y
    tumor_poly_coords = tumor_poly_coords - np.array([tile_x, tile_y])
    overlapping_tumor_poly = shapelyPolygon(tumor_poly_coords)
    # Create a psuedo mask
    psuedo_mask = poly2mask([overlapping_tumor_poly], (patch_size, patch_size))
    # Add it to the original mask
    mask = np.logical_or(mask, psuedo_mask)

    # If its inside tumor does this tumor patch actually contain any normal patches?
    tumor_poly = polygons['tumor'][tumor_poly_index]
    normal_patches_inside_tumor = get_common_interior_polygons(
        tumor_poly, polygons['normal'])

    # For all the normal patches, ensure
    # we set the mask to zero
    for index in normal_patches_inside_tumor:
        normal_poly = polygons['normal'][index]

        # What is the intersection portion of this normal polygon
        # with our patch of interest?
        common_area = normal_poly.intersection(patch_polygon)
        if common_area:
            normal_poly_coords = np.array(
                common_area.boundary.coords) - np.array([tile_x, tile_y])
            overlapping_normal_poly = shapelyPolygon(normal_poly_coords)
            psuedo_mask = poly2mask([overlapping_normal_poly],
                                    (patch_size, patch_size))
            # Get coordinates wherever this is non zero
            non_zero_coords = np.where(psuedo_mask > 0)
            # Add set these explicitly to zero
            mask[non_zero_coords] = 0
    return mask


def get_all_patches_from_slide(slide_path,
                               json_filepath=None,
                               filter_non_tissue=True,
                               patch_size=256,
                               saveto=None):
    """Extract a dataframe of all patches

    Parameters
    ----------
    slide_path: string
                Path to slide
    json_filepath: string
                   Path to annotation file
    filter_non_tissue: bool
                       Should remove tiles with no tissue?
    saveto: string
            Path to store output

    Returns
    -------
    tiles_df: pd.DataFrame
              A dataframe with the following columns:
                  slide_path - abs. path to slide
                  uid - slide filename
                  is_tissue - True/False
                  is_tumor - True/False
                  tile_loc - (row, col) coordinate of the tile

    """
    with WSIReader(slide_path, 40) as slide:
        thumbnail = slide.get_thumbnail(
            (int(slide.dimensions[0] / patch_size),
             int(slide.dimensions[1] / patch_size)))
        thumbnail_nrow = int(slide.width / patch_size)
        thumbnail_ncol = int(slide.height / patch_size)

    thumbnail_grey = np.array(thumbnail.convert('L'))  # convert to grayscale

    thresh = threshold_otsu(thumbnail_grey)
    binary = thumbnail_grey > thresh

    patches = pd.DataFrame(pd.DataFrame(binary).stack())
    patches.loc[:, 'is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)

    if json_filepath is not None:
        polygons = get_annotation_polygons(json_filepath)
        polymasked = get_approx_tumor_mask(polygons, thumbnail_nrow,
                                           thumbnail_ncol)

        patches_tumor = pd.DataFrame(pd.DataFrame(polymasked).stack())
        patches_tumor['is_tumor'] = patches_tumor[0] > 0
        patches_tumor.drop(0, axis=1, inplace=True)

        patches = pd.concat([patches, patches_tumor], axis=1)

    patches.loc[:, 'uid'] = os.path.basename(slide_path).replace('.tif', '')
    patches.loc[:, 'slide_path'] = os.path.abspath(slide_path)
    patches.loc[:, 'json_filepath'] = json_filepath
    if filter_non_tissue:
        patches = patches[patches.is_tissue ==
                          True]  # remove patches with no tissue
    patches['tile_loc'] = list(patches.index)
    patches.reset_index(inplace=True, drop=True)
    if saveto:
        patches.to_csv(saveto, sep='\t', index=False, header=True)
    return patches


def generate_tiles(samples,
                   batch_size=32,
                   patch_size=256,
                   num_classes=2,
                   convert_to_cat=True,
                   shuffle=True):
    """Generator function to yield image and mask tuples,

    Parameters
    ----------
    samples: DataFrame
             dataframe as obtained from get_all_patches_from_slide
    batch_size: int
                Batch size
    patch_size: int
                Patch size
    convert_to_cat: bool
                    Should convert to categorical
    shuffle: bool
             Should shuffle samples before yielding

    Returns
    -------
    Generator:
    X: tensor(float)
       Tensor of shape [batch_size, patch_size, patch_size, 3]
    Y: tensor(float)
       Tensor of shape [batch_size, patch_size, patch_size, NUM_CLASSES]


    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1)  # shuffle samples

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            images = []
            masks = []
            for _, batch_sample in batch_samples.iterrows():
                slide_contains_tumor = batch_sample['uid'].startswith('tumor')

                with WSIReader(batch_sample.slide_path, 40) as slide:
                    tiles = DeepZoomGenerator(
                        slide,
                        tile_size=patch_size,
                        overlap=0,
                        limit_bounds=False)
                    tile_loc = batch_sample.tile_loc#[::-1]
                    if isinstance(tile_loc, six.string_types):
                        tile_row, tile_col = eval(tile_loc)
                    else:
                        tile_row, tile_col = tile_loc
                    # the get_tile tuple required is (col, row)
                    img = tiles.get_tile(tiles.level_count - 1,
                                         (tile_col, tile_row))
                    (tile_x,
                     tile_y), tile_level, _ = tiles.get_tile_coordinates(
                         tiles.level_count - 1, (tile_col, tile_row))

                if slide_contains_tumor:
                    json_filepath = batch_sample['json_filepath']
                    polygons = get_annotation_polygons(json_filepath,
                                                       'shapely')
                    mask = create_tumor_mask_from_tile(tile_x, tile_y,
                                                       polygons, patch_size)
                else:
                    mask = np.zeros((patch_size, patch_size))

                images.append(np.array(img))
                masks.append(mask)

            X_train = np.array(images)
            y_train = np.array(masks)
            if convert_to_cat:
                y_train = mask_to_categorical(y_train, num_classes, patch_size)
            yield X_train, y_train
