import os
import numpy as np
import pandas as pd
from ..morphology.mask import get_common_interior_polygons
from .operations import get_annotation_polygons
from .operations import poly2mask
from .operations import translate_and_scale_polygon
from .operations import WSIReader

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from shapely.geometry import Point as shapelyPoint
from shapely.geometry import Polygon as shapelyPolygon


def get_approx_tumor_mask(polygons, thumbnail_nrow, thumbnail_ncol):
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
            if sample_intersection.geom_type != 'MultiPolygon':
                tumor_poly_coords = np.array(
                    sample_intersection.boundary.coords)
            else:
                tumor_poly_coords = []
                for p in sample_intersection:
                    tumor_poly_coords += p.boundary.coords
                tumor_poly_coords = np.array(tumor_poly_coords)

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
    #print(tumor_poly_index, tumor_poly_coords)
    #print(overlapping_tumor_poly.boundary.coords, tumor_poly_coords)
    #try:
    psuedo_mask = poly2mask([overlapping_tumor_poly], (patch_size, patch_size))
    #except:
    #    raise ValueError('{} | {}'.format(overlapping_tumor_poly.exterior,
    #                                      overlapping_tumor_poly.boundary.coords))
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
            psuedo_mask = poly2mask([overlapping_normal_poly], patch_size)
            # Get coordinates wherever this is non zero
            non_zero_coords = np.where(psuedo_mask > 0)
            # Add set these explicitly to zero
            mask[non_zero_coords] = 0
    return mask


def find_patches_from_slide(slide_path,
                            polygons,
                            add_tumor_patches=True,
                            filter_non_tissue=True):
    """Returns a dataframe of all patches in slide
    input: slide_path: path to WSI file
    output: samples: dataframe with the following columns:
        slide_path: path of slide
        is_tissue: sample contains tissue
        is_tumor: truth status of sample
        tile_loc: coordinates of samples in slide


    option: base_truth_dir: directory of truth slides
    option: filter_non_tissue: Remove samples no tissue detected
    """
    with WSIReader(slide_path) as slide:
        thumbnail = slide.get_thumbnail((int(slide.dimensions[0] / 256),
                                         int(slide.dimensions[1] / 256)))
        thumbnail_nrow = int(slide.width / 256)
        thumbnail_ncol = int(slide.height / 256)

    thumbnail_grey = np.array(thumbnail.convert('L'))  # convert to grayscale

    thresh = threshold_otsu(thumbnail_grey)
    binary = thumbnail_grey > thresh

    patches = pd.DataFrame(pd.DataFrame(binary).stack())
    patches.loc[:, 'is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)

    if add_tumor_patches:
        polymasked = get_approx_tumor_mask(polygons, thumbnail_nrow,
                                           thumbnail_ncol)

        patches_tumor = pd.DataFrame(pd.DataFrame(polymasked).stack())
        patches_tumor['is_tumor'] = patches_tumor[0] > 0
        patches_tumor.drop(0, axis=1, inplace=True)

        patches = pd.concat([patches, patches_tumor], axis=1)

    patches.loc[:, 'sample'] = os.path.basename(slide_path).replace('.tif', '')
    patches.loc[:, 'slide_path'] = slide_path
    if filter_non_tissue:
        patches = patches[patches.is_tissue ==
                          True]  # remove patches with no tissue
    patches['tile_loc'] = list(patches.index)
    patches.reset_index(inplace=True, drop=True)
    return patches
