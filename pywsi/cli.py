# -*- coding: utf-8 -*-
"""Console script for pywsi."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import six
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pywsi.io.operations import get_annotation_bounding_boxes
from pywsi.io.operations import get_annotation_polygons
from pywsi.io.operations import path_leaf
from pywsi.io.operations import read_as_rgb
from pywsi.io.operations import WSIReader
from pywsi.io.tiling import get_all_patches_from_slide
from pywsi.io.tiling import save_images_and_mask, generate_tiles, generate_tiles_fast
from pywsi.normalization import VahadaneNormalization

from pywsi.morphology.patch_extractor import TissuePatch
from pywsi.morphology.mask import get_common_interior_polygons
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from pywsi.segmentation import label_nuclei, summarize_region_properties
from pywsi.misc.parallel import ParallelExecutor
#from pywsi.deep_model.model import slide_level_map
#from pywsi.deep_model.random_forest import random_forest

from pywsi.misc import xmltojson
from scipy.misc import imsave

from collections import defaultdict
import joblib
from joblib import delayed
from joblib import parallel_backend
import numpy as np
from six import iteritems

import click
from shapely.geometry import Polygon as shapelyPolygon
from click_help_colors import HelpColorsGroup
import glob
from PIL import Image
click.disable_unicode_literals_warning = True
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
import pandas as pd
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.warnings.filterwarnings('ignore')

COLUMNS = [
    'area', 'bbox_area', 'compactness', 'convex_area', 'eccentricity',
    'equivalent_diameter', 'extent', 'fractal_dimension',
    'inertia_tensor_eigvals_1', 'inertia_tensor_eigvals_2',
    'major_axis_length', 'max_intensity', 'mean_intensity',
    'mean_intensity_entire_image', 'minor_axis_length', 'moments_central_1',
    'moments_central_10', 'moments_central_11', 'moments_central_12',
    'moments_central_13', 'moments_central_14', 'moments_central_15',
    'moments_central_16', 'moments_central_2', 'moments_central_3',
    'moments_central_4', 'moments_central_5', 'moments_central_6',
    'moments_central_7', 'moments_central_8', 'moments_central_9',
    'moments_hu_1', 'moments_hu_2', 'moments_hu_3', 'moments_hu_4',
    'moments_hu_5', 'moments_hu_6', 'moments_hu_7', 'nuclei',
    'nuclei_intensity_over_entire_image', 'orientation', 'perimeter',
    'solidity', 'texture', 'total_nuclei_area', 'total_nuclei_area_ratio'
]


@click.group(
    cls=HelpColorsGroup,
    help_headers_color='yellow',
    help_options_color='green')
def cli():
    """pywsi: tool for processing WSIs"""
    pass


@cli.command(
    'create-tissue-masks',
    context_settings=CONTEXT_SETTINGS,
    help='Extract tissue masks')
@click.option(
    '--indir', help='Root directory with all tumor WSIs', required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_tissue_masks_cmd(indir, level, savedir):
    """Extract tissue only patches from tumor WSIs.
    """
    tumor_wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    for tumor_wsi in tqdm(tumor_wsis):
        wsi = WSIReader(tumor_wsi, 40)
        tissue_patch = TissuePatch(wsi, level=level)
        uid = wsi.uid.replace('.tif', '')
        out_file = os.path.join(savedir, 'level_{}'.format(level),
                                uid + '_TissuePatch.npy')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.save(out_file, tissue_patch.otsu_thresholded)


@cli.command(
    'create-annotation-masks',
    context_settings=CONTEXT_SETTINGS,
    help='Extract annotation masks')
@click.option(
    '--indir', help='Root directory with all tumor WSIs', required=True)
@click.option('--jsondir', help='Root directory with all jsons', required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_annotation_masks_cmd(indir, jsondir, level, savedir):
    """Extract annotation patches

    We assume the masks have already been generated at level say x.
    We also assume the files are arranged in the following heirerachy:

        raw data (indir): tumor_wsis/tumor001.tif
        json data (jsondir): tumor_jsons/tumor001.json

    """
    tumor_wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    for tumor_wsi in tqdm(tumor_wsis):
        wsi = WSIReader(tumor_wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        json_filepath = os.path.join(jsondir, uid + '.json')
        if not os.path.exists(json_filepath):
            print('Skipping {} as annotation json not found'.format(uid))
            continue
        out_dir = os.path.join(savedir, 'level_{}'.format(level))
        wsi.annotation_masked(
            json_filepath=json_filepath, level=level, savedir=out_dir)


@cli.command(
    'extract-tumor-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Extract tumor patches from tumor WSIs')
@click.option(
    '--indir', help='Root directory with all tumor WSIs', required=True)
@click.option(
    '--annmaskdir',
    help='Root directory with all annotation mask WSIs',
    required=True)
@click.option(
    '--tismaskdir',
    help='Root directory with all annotation mask WSIs',
    required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--patchsize',
    type=int,
    default=128,
    help='Patch size which to extract patches')
@click.option(
    '--stride', type=int, default=128, help='Stride to generate next patch')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
@click.option(
    '--threshold',
    help='Threshold for a cell to be called tumor',
    default=0,
    type=int)
def extract_tumor_patches_cmd(indir, annmaskdir, tismaskdir, level, patchsize,
                              stride, savedir, threshold):
    """Extract tumor only patches from tumor WSIs.

    We assume the masks have already been generated at level say x.
    We also assume the files are arranged in the following heirerachy:

        raw data (indir): tumor_wsis/tumor001.tif
        masks (maskdir): tumor_masks/level_x/tumor001_AnnotationTumorMask.npy';
                         tumor_masks/level_x/tumor001_AnnotationNormalMask.npy';

    We create the output in a similar fashion:
        output (outdir): patches/tumor/level_x/tumor001_xcenter_ycenter.png


    Strategy:

        1. Load tumor annotated masks
        2. Load normal annotated masks
        3. Do subtraction tumor-normal to ensure only tumor remains.

        Truth table:

            tumor_mask  normal_mask  tumour_for_sure
                1           0            1
                1           1            0
                1           1            0
                0           1            0
    """
    tumor_wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)

    # Assume that we want to generate these patches at level 0
    # So in order to ensure stride at a lower level
    # this needs to be discounted
    #stride = int(patchsize / (2**level))
    stride = min(int(patchsize / (2**level)), 4)
    for tumor_wsi in tqdm(tumor_wsis):
        last_used_x = None
        last_used_y = None
        wsi = WSIReader(tumor_wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        filepath = os.path.join(annmaskdir, 'level_{}'.format(level),
                                uid + '_AnnotationColored.npy')
        if not os.path.exists(filepath):
            print('Skipping {} as mask not found'.format(uid))
            continue
        normal_mask = np.load(
            os.path.join(annmaskdir, 'level_{}'.format(level),
                         uid + '_AnnotationNormalMask.npy'))
        tumor_mask = np.load(
            os.path.join(annmaskdir, 'level_{}'.format(level),
                         uid + '_AnnotationTumorMask.npy'))
        tissue_mask = np.load(
            os.path.join(tismaskdir, 'level_{}'.format(level),
                         uid + '_TissuePatch.npy'))

        colored_patch = np.load(
            os.path.join(annmaskdir, 'level_{}'.format(level),
                         uid + '_AnnotationColored.npy'))
        subtracted_mask = tumor_mask * 1 - normal_mask * 1
        subtracted_mask[np.where(subtracted_mask < 0)] = 0
        subtracted_mask = np.logical_and(subtracted_mask, tissue_mask)
        x_ids, y_ids = np.where(subtracted_mask)
        for x_center, y_center in zip(x_ids, y_ids):
            out_file = '{}/level_{}/{}_{}_{}_{}.png'.format(
                savedir, level, uid, x_center, y_center, patchsize)
            x_topleft = int(x_center - patchsize / 2)
            y_topleft = int(y_center - patchsize / 2)
            x_topright = x_topleft + patchsize
            y_bottomright = y_topleft + patchsize
            #print((x_topleft, x_topright, y_topleft, y_bottomright))
            mask = subtracted_mask[x_topleft:x_topright, y_topleft:
                                   y_bottomright]
            # Feed only complete cancer cells
            # Feed if more thatn 50% cells are cancerous!
            if threshold <= 0:
                threshold = 0.5 * (patchsize * patchsize)
            if np.sum(mask) > threshold:
                if last_used_x is None:
                    last_used_x = x_center
                    last_used_y = y_center
                    diff_x = stride
                    diff_y = stride
                else:
                    diff_x = np.abs(x_center - last_used_x)
                    diff_y = np.abs(y_center - last_used_y)
                if diff_x >= stride and diff_y >= stride:
                    patch = colored_patch[x_topleft:x_topright, y_topleft:
                                          y_bottomright, :]
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    img = Image.fromarray(patch)
                    img.save(out_file)
                    last_used_x = x_center
                    last_used_y = y_center


@cli.command(
    'extract-normal-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Extract normal patches from tumor WSIs')
@click.option(
    '--indir', help='Root directory with all tumor WSIs', required=True)
@click.option(
    '--annmaskdir',
    help='Root directory with all annotation mask WSIs',
    required=False)
@click.option(
    '--tismaskdir',
    help='Root directory with all annotation mask WSIs',
    required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--patchsize',
    type=int,
    default=128,
    help='Patch size which to extract patches')
@click.option(
    '--stride', type=int, default=128, help='Stride to generate next patch')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_normal_patches_cmd(indir, annmaskdir, tismaskdir, level, patchsize,
                               stride, savedir):
    """Extract tumor only patches from tumor WSIs.

    We assume the masks have already been generated at level say x.
    We also assume the files are arranged in the following heirerachy:

        raw data (indir): tumor_wsis/tumor001.tif
        masks (maskdir): tumor_masks/level_x/tumor001_AnnotationTumorMask.npy';
                         tumor_masks/level_x/tumor001_AnnotationNormalMask.npy';

    We create the output in a similar fashion:
        output (outdir): patches/tumor/level_x/tumor001_xcenter_ycenter.png


    Strategy:

        1. Load tumor annotated masks
        2. Load normal annotated masks
        3. Do subtraction tumor-normal to ensure only tumor remains.

        Truth table:

            tumor_mask  normal_mask  tumour_for_sure
                1           0            1
                1           1            0
                1           1            0
                0           1            0
    """
    all_wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=True)

    # Assume that we want to generate these patches at level 0
    # So in order to ensure stride at a lower level
    # this needs to be discounted
    stride = min(int(patchsize / (2**level)), 4)
    for wsi in tqdm(all_wsis):
        last_used_x = None
        last_used_y = None
        wsi = WSIReader(wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        tissue_mask = np.load(
            os.path.join(tismaskdir, 'level_{}'.format(level),
                         uid + '_TissuePatch.npy'))
        if 'normal' in uid:
            # Just extract based on tissue patches
            x_ids, y_ids = np.where(tissue_mask)
            subtracted_mask = tissue_mask
            colored_patch = wsi.get_patch_by_level(0, 0, level)
        elif 'tumor' in uid or 'test' in uid:
            if not os.path.isfile(
                    os.path.join(annmaskdir, 'level_{}'.format(level),
                                 uid + '_AnnotationNormalMask.npy')):
                print('Skipping {}'.format(uid))
                continue
            normal_mask = np.load(
                os.path.join(annmaskdir, 'level_{}'.format(level),
                             uid + '_AnnotationNormalMask.npy'))
            tumor_mask = np.load(
                os.path.join(annmaskdir, 'level_{}'.format(level),
                             uid + '_AnnotationTumorMask.npy'))
            colored_patch = np.load(
                os.path.join(annmaskdir, 'level_{}'.format(level),
                             uid + '_AnnotationColored.npy'))

            subtracted_mask = normal_mask * 1 - tumor_mask * 1
            subtracted_mask[np.where(subtracted_mask < 0)] = 0
            subtracted_mask = np.logical_and(subtracted_mask, tissue_mask)
            x_ids, y_ids = np.where(subtracted_mask)
        for x_center, y_center in zip(x_ids, y_ids):
            out_file = '{}/level_{}/{}_{}_{}_{}.png'.format(
                savedir, level, uid, x_center, y_center, patchsize)
            x_topleft = int(x_center - patchsize / 2)
            y_topleft = int(y_center - patchsize / 2)
            x_topright = x_topleft + patchsize
            y_bottomright = y_topleft + patchsize
            mask = subtracted_mask[x_topleft:x_topright, y_topleft:
                                   y_bottomright]
            # Feed if more thatn 50% masks are positive
            if np.sum(mask) > 0.5 * (patchsize * patchsize):
                if last_used_x is None:
                    last_used_x = x_center
                    last_used_y = y_center
                    diff_x = stride
                    diff_y = stride
                else:
                    diff_x = np.abs(x_center - last_used_x)
                    diff_y = np.abs(y_center - last_used_y)
                if diff_x >= stride and diff_y >= stride:
                    patch = colored_patch[x_topleft:x_topright, y_topleft:
                                          y_bottomright, :]
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    img = Image.fromarray(patch)
                    img.save(out_file)
                    last_used_x = x_center
                    last_used_y = y_center


@cli.command(
    'patches-from-coords',
    context_settings=CONTEXT_SETTINGS,
    help='Extract patches from coordinates file')
@click.option('--indir', help='Root directory with all WSIs', required=True)
@click.option('--csv', help='Path to csv with coordinates', required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--patchsize',
    type=int,
    default=128,
    help='Patch size which to extract patches')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_patches_from_coords_cmd(indir, csv, level, patchsize, savedir):
    """Extract patches from coordinates file at a particular level.

    Assumption: Coordinates are assumed to be provided at level 0.
    """
    patches_to_extract = defaultdict(list)
    with open(csv) as fh:
        for line in fh:
            try:
                filename, x0, y0 = line.split(',')
            except:
                splitted = line.split('_')
                # test files have name like test_001
                if len(splitted) == 5:
                    fileprefix, fileid, x0, y0, _ = splitted
                    filename = '{}_{}'.format(fileprefix, fileid)
                elif len(splitted) == 4:
                    filename, x0, y0, _ = splitted
                else:
                    raise RuntimeError(
                        'Unable to find parsable format. Mustbe filename,x0,y-'
                    )
                # other files have name like normal001

            filename = filename.lower()
            x0 = int(x0)
            y0 = int(y0)
            patches_to_extract[filename].append((x0, y0))

    for filename, coordinates in tqdm(patches_to_extract.items()):
        if 'normal' in filename:
            filepath = os.path.join(indir, 'normal', filename + '.tif')
        elif 'tumor' in filename:
            filepath = os.path.join(indir, 'tumor', filename + '.tif')
        elif 'test' in filename:
            filepath = os.path.join(indir, filename + '.tif')
        else:
            raise RuntimeError('Malformed filename?: {}'.format(filename))
        wsi = WSIReader(filepath, 40)
        uid = wsi.uid.replace('.tif', '')
        for x0, y0 in coordinates:
            patch = wsi.get_patch_by_level(x0, y0, level, patchsize)
            out_file = '{}/level_{}/{}_{}_{}_{}.png'.format(
                savedir, level, uid, x0, y0, patchsize)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            img = Image.fromarray(patch)
            img.save(out_file)


@cli.command(
    'extract-test-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Extract patches from  testing dataset')
@click.option('--indir', help='Root directory with all WSIs', required=True)
@click.option(
    '--tismaskdir',
    help='Root directory with all annotation mask WSIs',
    required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--patchsize',
    type=int,
    default=128,
    help='Patch size which to extract patches')
@click.option(
    '--stride',
    default=64,
    help='Slide windows by this much to get the next [atj]',
    required=True)
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_test_patches_cmd(indir, tismaskdir, level, patchsize, stride,
                             savedir):
    wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    for wsi in tqdm(wsis):
        last_used_y = None
        last_used_x = None
        wsi = WSIReader(wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        tissue_mask = np.load(
            os.path.join(tismaskdir, 'level_{}'.format(level),
                         uid + '_TissuePatch.npy'))

        x_ids, y_ids = np.where(tissue_mask)
        for x_center, y_center in zip(x_ids, y_ids):
            out_file = '{}/level_{}/{}_{}_{}_{}.png'.format(
                savedir, level, uid, x_center, y_center, patchsize)
            x_topleft = int(x_center - patchsize / 2)
            y_topleft = int(y_center - patchsize / 2)
            x_topright = x_topleft + patchsize
            y_bottomright = y_topleft + patchsize
            mask = tissue_mask[x_topleft:x_topright, y_topleft:y_bottomright]
            if np.sum(mask) > 0.5 * (patchsize * patchsize):
                if last_used_x is None:
                    last_used_x = x_center
                    last_used_y = y_center
                    diff_x = stride
                    diff_y = stride
                else:
                    diff_x = np.abs(x_center - last_used_x)
                    diff_y = np.abs(y_center - last_used_y)
                if diff_x >= stride or diff_y >= stride:
                    colored_patch = wsi.get_patch_by_level(0, 0, level)
                    patch = colored_patch[x_topleft:x_topright, y_topleft:
                                          y_bottomright, :]
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    img = Image.fromarray(patch)
                    img.save(out_file)
                    last_used_x = x_center
                    last_used_y = y_center


@cli.command(
    'estimate-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Estimate number of extractable tumor patches from tumor WSIs')
@click.option(
    '--indir', help='Root directory with all tumor WSIs', required=True)
@click.option('--jsondir', help='Root directory with all jsons', required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--patchsize',
    type=int,
    default=128,
    help='Patch size which to extract patches')
@click.option(
    '--stride', type=int, default=128, help='Stride to generate next patch')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def estimate_patches_cmd(indir, jsondir, level, patchsize, stride, savedir):
    all_wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    out_dir = os.path.join(savedir, 'level_{}'.format(level))
    os.makedirs(out_dir, exist_ok=True)
    for wsi in tqdm(all_wsis):
        wsi = WSIReader(wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        json_filepath = os.path.join(jsondir, uid + '.json')
        if not os.path.exists(json_filepath):
            print('Skipping {} as annotation json not found'.format(uid))
            continue
        bounding_boxes = get_annotation_bounding_boxes(json_filepath)
        polygons = get_annotation_polygons(json_filepath)
        tumor_bb = bounding_boxes['tumor']
        normal_bb = bounding_boxes['normal']

        normal_polygons = polygons['normal']
        tumor_polygons = polygons['tumor']
        polygons_dict = {'normal': normal_polygons, 'tumor': tumor_polygons}
        rectangles_dict = {'normal': normal_bb, 'tumor': tumor_bb}
        for polygon_key, polygons in iteritems(polygons_dict):
            bb = rectangles_dict[polygon_key]
            to_write = ''
            with open(os.path.join(savedir, '{}.txt', 'w')) as fh:
                for rectangle, polygon in zip(bb, polygons):
                    """
                    Sample points from rectangle. We will assume we are sampling the
                    centers of our patch. So if we sample x_center, y_center
                    from this rectangle, we need to ensure (x_center +/- patchsize/2, y_center +- patchsize/2)
                    lie inside the polygon
                    """
                    xmin, ymax = rectangle['top_left']
                    xmax, ymin = rectangle['bottom_right']
                    path = polygon.get_path()
                    for x_center in np.arange(xmin, xmax, patchsize):
                        for y_center in np.arange(ymin, ymax, patchsize):
                            x_topleft = int(x_center - patchsize / 2)
                            y_topleft = int(y_center - patchsize / 2)
                            x_bottomright = x_topleft + patchsize
                            y_bottomright = y_topleft + patchsize

                            if path.contains_points([(x_topleft, y_topleft),
                                                     (x_bottomright,
                                                      y_bottomright)]).all():
                                to_write = '{}_{}_{}_{}\n'.format(
                                    uid, x_center, y_center, patchsize)
                                fh.write(to_write)


def process_wsi(data):
    wsi, jsondir, patchsize, stride, level, dirs, write_image = data
    wsi = WSIReader(wsi, 40)
    uid = wsi.uid.replace('.tif', '')
    scale_factor = wsi.get_level_scale_factor(level)
    json_filepath = os.path.join(jsondir, uid + '.json')
    if not os.path.isfile(json_filepath):
        return
    boxes = get_annotation_bounding_boxes(json_filepath)
    polygons = get_annotation_polygons(json_filepath)

    polygons_to_exclude = {'tumor': [], 'normal': []}

    for polygon in polygons['tumor']:
        # Does this have any of the normal polygons inside it?
        polygons_to_exclude['tumor'].append(
            get_common_interior_polygons(polygon, polygons['normal']))

    for polygon in polygons['normal']:
        # Does this have any of the tumor polygons inside it?
        polygons_to_exclude['normal'].append(
            get_common_interior_polygons(polygon, polygons['tumor']))

    for polygon_key in polygons.keys():
        last_used_x = None
        last_used_y = None
        annotated_polygons = polygons[polygon_key]
        annotated_boxes = boxes[polygon_key]

        # iterate through coordinates in the bounding rectangle
        # tand check if they overlap with any other annoations and
        # if not fetch a patch at that coordinate from the wsi
        annotation_index = 0
        for annotated_polygon, annotated_box in zip(annotated_polygons,
                                                    annotated_boxes):
            annotation_index += 1
            minx, miny = annotated_box['top_left']
            maxx, miny = annotated_box['top_right']

            maxx, maxy = annotated_box['bottom_right']
            minx, maxy = annotated_box['bottom_left']

            width = int(maxx) - int(minx)
            height = int(maxy) - int(miny)
            #(minx, miny), width, height = annotated_box['top_left'], annotated_box['top'].get_xy()
            # Should scale?
            # No. Do not scale here as the patch is always
            # fetched from things at level0
            minx = int(minx)  # * scale_factor)
            miny = int(miny)  # * scale_factor)
            maxx = int(maxx)  # * scale_factor)
            maxy = int(maxy)  # * scale_factor)

            width = int(width * scale_factor)
            height = int(height * scale_factor)

            annotated_polygon = np.array(annotated_polygon.get_xy())

            annotated_polygon = annotated_polygon * scale_factor

            # buffer ensures the resulting polygon is clean
            # http://toblerity.org/shapely/manual.html#object.buffer
            try:
                annotated_polygon_scaled = shapelyPolygon(
                    np.round(annotated_polygon).astype(int)).buffer(0)
            except:
                warnings.warn(
                    'Skipping creating annotation index {} for {}'.format(
                        annotation_index, uid))
                continue
            assert annotated_polygon_scaled.is_valid, 'Found invalid annotated polygon: {} {}'.format(
                uid,
                shapelyPolygon(annotated_polygon).is_valid)
            for x_left in np.arange(minx, maxx, 1):
                for y_top in np.arange(miny, maxy, 1):
                    x_right = x_left + patchsize
                    y_bottom = y_top + patchsize
                    if last_used_x is None:
                        last_used_x = x_left
                        last_used_y = y_top
                        diff_x = stride
                        diff_y = stride
                    else:
                        diff_x = np.abs(x_left - last_used_x)
                        diff_y = np.abs(y_top - last_used_y)
                    #print(last_used_x, last_used_y, x_left, y_top, diff_x, diff_y)
                    if diff_x <= stride or diff_y <= stride:
                        continue
                    else:
                        last_used_x = x_left
                        last_used_y = y_top
                    patch_polygon = shapelyPolygon(
                        [(x_left, y_top), (x_right, y_top),
                         (x_right, y_bottom), (x_left, y_bottom)]).buffer(0)
                    assert patch_polygon.is_valid, 'Found invalid polygon: {}_{}_{}'.format(
                        uid, x_left, y_top)
                    try:
                        is_inside = annotated_polygon_scaled.contains(
                            patch_polygon)
                    except:
                        # Might raise an exception when the two polygons
                        # are the same
                        warnings.warn(
                            'Skipping: {}_{}_{}_{}.png | Equals: {} | Almost equals: {}'.
                            format(uid, x_left, y_top, patchsize),
                            annotated_polygon_scaled.equals(patch_polygon),
                            annotated_polygon_scaled.almost_equals(
                                patch_polygon))
                        continue

                    if write_image:
                        out_file = os.path.join(
                            dirs[polygon_key], '{}_{}_{}_{}.png'.format(
                                uid, x_left, y_top, patchsize))
                        patch = wsi.get_patch_by_level(x_left, y_top, level,
                                                       patchsize)
                        os.makedirs(os.path.dirname(out_file), exist_ok=True)
                        img = Image.fromarray(patch)
                        img.save(out_file)
                    else:
                        # Just write the coordinates
                        to_write = '{}_{}_{}_{}\n'.format(
                            uid, x_left, y_top, patchsize)
                        out_file = os.path.join(dirs[polygon_key],
                                                '{}.txt'.format(polygon_key))
                        with open(out_file, 'a') as fh:
                            fh.write(to_write)


@cli.command(
    'extract-test-both-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Extract both normal and tumor patches from tissue masks')
@click.option(
    '--indir', help='Root directory with all test WSIs', required=True)
@click.option(
    '--patchsize',
    type=int,
    default=128,
    help='Patch size which to extract patches')
@click.option(
    '--stride', type=int, default=128, help='Stride to generate next patch')
@click.option('--jsondir', help='Root directory with all jsons', required=True)
@click.option(
    '--level',
    type=int,
    help='Level at which to extract patches',
    required=True)
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
@click.option('--write_image', help='Should output images', is_flag=True)
def extract_test_both_cmd(indir, patchsize, stride, jsondir, level, savedir,
                          write_image):
    """Extract tissue only patches from tumor WSIs.
    """
    wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    out_dir = os.path.join(savedir, 'level_{}'.format(level))
    normal_dir = os.path.join(out_dir, 'normal')
    tumor_dir = os.path.join(out_dir, 'tumor')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(tumor_dir, exist_ok=True)
    dirs = {'normal': normal_dir, 'tumor': tumor_dir}

    total_wsi = len(wsis)
    data = [(wsi, jsondir, patchsize, stride, level, dirs, write_image)
            for wsi in wsis]
    with tqdm(total=total_wsi) as pbar:
        with Pool(processes=16) as p:
            for i, _ in enumerate(p.imap_unordered(process_wsi, data)):
                #print(i / total_wsi * 100)
                pbar.update()
        #    for i, wsi in tqdm(enumerate(list(wsis))):
        #        process_wsi(wsi)
        #        pbar.update()


def process_segmentation(data):
    """
    Parameters
    ----------
    data: tuple
          (png_location, tsv_outpath)

    """

    png, saveto = data
    patch = read_as_rgb(png)
    region_properties, _ = label_nuclei(patch, draw=False)
    summary = summarize_region_properties(region_properties, patch)
    df = pd.DataFrame([summary])
    df.to_csv(saveto, index=False, header=True, sep='\t')


@cli.command(
    'segment',
    context_settings=CONTEXT_SETTINGS,
    help='Performs segmentation and extract-features')
@click.option('--indir', help='Root directory with all pngs', required=True)
@click.option('--outdir', help='Output directory to out tsv', required=True)
def segementation_cmd(indir, outdir):
    """Perform segmentation and store the tsvs
    """
    list_of_pngs = list(glob.glob(indir + '/*.png'))
    data = []
    for f in list_of_pngs:
        tsv = f.replace(os.path.dirname(f), outdir).replace('.png', '.tsv')
        if not os.path.isfile(tsv):
            data.append((f, tsv))
        elif os.stat(tsv).st_size == 0:
            data.appen((f, tsv))

    os.makedirs(outdir, exist_ok=True)
    with tqdm(total=len(data)) as pbar:
        with Pool(processes=16) as p:
            for i, _ in enumerate(
                    p.imap_unordered(process_segmentation, data)):
                pbar.update()


def _process_patches_df(data):
    slide_path, json_filepath, patch_size, saveto = data
    df = get_all_patches_from_slide(
        slide_path,
        json_filepath=json_filepath,
        filter_non_tissue=True,
        patch_size=patch_size,
        saveto=saveto)
    return df


@cli.command(
    'patches-df',
    context_settings=CONTEXT_SETTINGS,
    help='Extract all patches summarized as dataframes')
@click.option(
    '--indir', help='Root directory with all tumor WSIs', required=True)
@click.option('--jsondir', help='Root directory with all jsons')
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_mask_df_cmd(indir, jsondir, patchsize, savedir):
    """Extract tissue only patches from tumor WSIs.
    """
    wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    data = []
    df = pd.DataFrame()
    for wsi in wsis:
        basename = path_leaf(wsi).replace('.tif', '')
        if jsondir:
            json_filepath = os.path.join(jsondir, basename + '.json')
        else:
            json_filepath = None
        if not os.path.isfile(json_filepath):
            json_filepath = None
        saveto = os.path.join(savedir, basename + '.tsv')
        data.append((wsi, json_filepath, patchsize, saveto))
    os.makedirs(savedir, exist_ok=True)
    with tqdm(total=len(wsis)) as pbar:
        with Pool(processes=16) as p:
            for i, temp_df in enumerate(
                    p.imap_unordered(_process_patches_df, data)):
                df = pd.concat([df, temp_df])
                pbar.update()
    if 'is_tumor' in df.columns:
        df = df.sort_values(by=['uid', 'is_tumor'])
    else:
        df['is_tumor'] = False
        df = df.sort_values(by=['uid'])

    df.to_csv(
        os.path.join(savedir, 'master_df.tsv'),
        sep='\t',
        index=False,
        header=True)


@cli.command(
    'tif-to-df',
    context_settings=CONTEXT_SETTINGS,
    help='Extract all patches summarized as dataframes from one WSI')
@click.option('--tif', help='Tif', required=True)
@click.option('--jsondir', help='Root directory with all jsons')
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def extract_df_from_tif_cmd(tif, jsondir, patchsize, savedir):
    """Extract tissue only patches from tumor WSIs.
    """
    basename = path_leaf(tif).replace('.tif', '')
    if jsondir:
        json_filepath = os.path.abspath(
            os.path.join(jsondir, basename + '.json'))
    else:
        json_filepath = None
    if not os.path.isfile(json_filepath):
        json_filepath = None
    saveto = os.path.join(savedir, basename + '.tsv')
    df = get_all_patches_from_slide(
        tif,
        json_filepath=json_filepath,
        filter_non_tissue=False,
        patch_size=patchsize,
        saveto=saveto)


@cli.command(
    'patch-and-mask',
    context_settings=CONTEXT_SETTINGS,
    help='Extract all patches and their mask from patches dataframes')
@click.option('--df', help='Path to dataframe', required=True)
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
@click.option('--savedf', help='Save edited dataframe to', required=True)
def extract_patch_mask_cmd(df, patchsize, savedir, savedf):
    """Extract tissue only patches from tumor WSIs.
    """
    assert not os.path.isfile(savedf)
    df = pd.read_table(df)
    df_copy = df.copy()
    df_copy['img_path'] = None
    df_copy['mask_path'] = None
    df['savedir'] = savedir
    df['patch_size'] = patchsize
    records = df.reset_index().to_dict('records')

    with tqdm(total=len(df.index)) as pbar:
        with Pool(processes=8) as p:
            for idx, img_path, mask_path in p.imap_unordered(
                    save_images_and_mask, records):
                df_copy.loc[idx, 'img_path'] = img_path
                df_copy.loc[idx, 'mask_path'] = mask_path
                pbar.update()
    df_copy.to_csv(savedf, sep='\t', index=False, header=True)


def process_segmentation_both(data):
    """
    Parameters
    ----------
    data: tuple
          (png_location, tsv_outpath)

    """

    is_tissue, is_tumor, pickle_file, savetopng, savetodf = data
    if not is_tissue:
        df = pd.DataFrame()
        df['is_tumor'] = is_tumor
        df['is_tissue'] = is_tissue
        return df

    patch = joblib.load(pickle_file)
    region_properties, _ = label_nuclei(
        patch, draw=False)  #savetopng=savetopng)
    summary = summarize_region_properties(region_properties, patch)
    df = pd.DataFrame([summary])
    df['is_tumor'] = is_tumor
    df.to_csv(savetodf, index=False, header=True, sep='\t')
    return df


@cli.command(
    'segment-from-df',
    context_settings=CONTEXT_SETTINGS,
    help='Segment from df')
@click.option('--df', help='Path to dataframe', required=True)
@click.option('--finaldf', help='Path to dataframe', required=True)
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def process_df_cmd(df, finaldf, savedir):
    df = pd.read_table(df)
    df['img_path'] = None
    df['mask_path'] = None
    modified_df = pd.DataFrame()
    os.makedirs(savedir, exist_ok=True)
    tile_loc = df.tile_loc.astype(str)
    tile_loc = tile_loc.str.replace(' ', '').str.replace(')', '').str.replace(
        '(', '')

    df[['row', 'col']] = tile_loc.str.split(',', expand=True)
    df['segmented_png'] = savedir + '/' + df[['uid', 'row', 'col']].apply(
        lambda x: '_'.join(x.values.tolist()), axis=1) + '.segmented.png'
    df['segmented_tsv'] = savedir + '/' + df[['uid', 'row', 'col']].apply(
        lambda x: '_'.join(x.values.tolist()), axis=1) + '.segmented.tsv'
    with tqdm(total=len(df.index)) as pbar:
        with Pool(processes=8) as p:
            for i, temp_df in enumerate(
                    p.imap_unordered(
                        process_segmentation_both, df[[
                            'is_tissue', 'is_tumor', 'img_path',
                            'segmented_png', 'segmented_tsv'
                        ]].values.tolist())):
                modified_df = pd.concat([modified_df, temp_df])
                pbar.update()

    modified_df.to_csv(finaldf, sep='\t', index=False, header=True)


def process_segmentation_fixed(batch_sample):
    patch_size = batch_sample['patch_size']
    savedir = os.path.abspath(batch_sample['savedir'])
    tile_loc = batch_sample['tile_loc']  #[::-1]
    segmentedmethod = batch_sample['segmented_method']
    if isinstance(tile_loc, six.string_types):
        tile_row, tile_col = eval(tile_loc)
    else:
        tile_row, tile_col = tile_loc
    segmented_img_path = os.path.join(
        savedir, batch_sample['uid'] + '_{}_{}.segmented.png'.format(
            tile_row, tile_col))
    segmented_tsv_path = os.path.join(
        savedir, batch_sample['uid'] + '_{}_{}.segmented_summary.tsv'.format(
            tile_row, tile_col))
    if os.path.isfile(segmented_img_path) and os.path.isfile(
            segmented_tsv_path):
        df = pd.read_table(segmented_tsv_path)
        return batch_sample[
            'index'], segmented_img_path, segmented_tsv_path, df

        # the get_tile tuple required is (col, row)
    if not os.path.isfile(batch_sample['img_path']):
        save_images_and_mask(batch_sample)
    patch = joblib.load(batch_sample['img_path'])
    region_properties, _ = label_nuclei(
        patch, draw=False,
        normalization=segmentedmethod)  #, savetopng=segmented_img_path)
    summary = summarize_region_properties(region_properties, patch)

    df = pd.DataFrame([summary])
    try:
        df['is_tumor'] = batch_sample['is_tumor']
    except KeyError:
        # Must be from a normal sample
        df['is_tumor'] = False
    df['is_tissue'] = batch_sample['is_tissue']

    df.to_csv(segmented_tsv_path, index=False, header=True, sep='\t')
    return batch_sample['index'], segmented_img_path, segmented_tsv_path, df


@cli.command(
    'segment-from-df-fast',
    context_settings=CONTEXT_SETTINGS,
    help='Segment from df')
@click.option('--df', help='Path to dataframe', required=True)
@click.option('--finaldf', help='Path to dataframe', required=True)
@click.option(
    '--segmethod',
    help='Path to dataframe',
    default=None,
    type=click.Choice(['None', 'vahadane', 'macenko', 'xu']))
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
@click.option(
    '--ncpu', type=int, default=1, help='Patch size which to extract patches')
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
def process_df_cmd_fast(df, finaldf, segmethod, savedir, ncpu, patchsize):
    savedir = os.path.abspath(savedir)
    df_main = pd.read_table(df)
    df = df_main.copy()
    df['savedir'] = savedir
    df['patch_size'] = patchsize
    df['segmented_png'] = None
    df['segmented_tsv'] = None
    df['segmented_method'] = segmethod

    modified_df = pd.DataFrame()
    os.makedirs(savedir, exist_ok=True)
    df_reset_index = df.reset_index()
    df_subset = df_reset_index[df_reset_index.is_tissue == True]
    records = df_subset.to_dict('records')
    with tqdm(total=len(df_subset.index)) as pbar:
        if ncpu > 1:
            with Pool(processes=ncpu) as p:
                for idx, segmented_png, segmented_tsv, summary_df in p.imap_unordered(
                        process_segmentation_fixed, records):
                    df.loc[idx, 'segmented_png'] = segmented_png
                    df.loc[idx, 'segmented_tsv'] = segmented_tsv
                    modified_df = pd.concat([modified_df, summary_df])
                    modified_df['index'] = idx
                    pbar.update()
        else:
            for idx, row in df.iterrows():
                row['index'] = idx
                _, segmented_png, segmented_tsv, summary_df = process_segmentation_fixed(
                    row)
                df.loc[idx, 'segmented_png'] = segmented_png
                df.loc[idx, 'segmented_tsv'] = segmented_tsv
                modified_df = pd.concat([modified_df, summary_df])
                modified_df['index'] = idx
                pbar.update()

    modified_df = modified_df.set_index('index')
    modified_df.to_csv(finaldf, sep='\t', index=False, header=True)
    df.to_csv(
        finaldf.replace('.tsv', '') + '.segmented.tsv',
        sep='\t',
        index=False,
        header=True)


def predict_batch_from_model(patches, model):
    """Predict which pixels are tumor.

    input: patch: `batch_size`x256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    predictions = model.predict(patches)
    predictions = predictions[:, :, :, 1]
    return predictions


def predict_from_model(patch, model):
    """Predict which pixels are tumor.

    input: patch: 256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """

    prediction = model.predict(patch.reshape(1, 256, 256, 3))
    prediction = prediction[:, :, :, 1].reshape(256, 256)
    return prediction


@cli.command(
    'heatmap',
    context_settings=CONTEXT_SETTINGS,
    help='Create tumor probability heatmap')
@click.option('--indir', help='Root directory with all WSIs', required=True)
@click.option(
    '--jsondir', help='Root directory with all jsons', required=False)
@click.option(
    '--imgmaskdir',
    help='Directory where the patches and mask are stored',
    default=
    '/Z/personal-folders/interns/saket/github/pywsi/data/patch_img_and_mask/')
@click.option(
    '--modelf',
    help='Root directory with all jsons',
    default=
    '/Z/personal-folders/interns/saket/github/pywsi/notebooks/weights-improvement-12-0.98.hdf'
)
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
@click.option('--gpu', type=str, default='0', help='Which GPU to use?')
@click.option(
    '--gpumemfrac',
    type=float,
    default=1,
    help='Fraction of GPU memory to use')
def create_tumor_map_cmd(indir, jsondir, imgmaskdir, modelf, patchsize,
                         savedir, gpu, gpumemfrac):
    """Extract probability maps for a WSI
    """
    batch_size = 32
    img_mask_dir = imgmaskdir
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpumemfrac
    config.gpu_options.visible_device_list = gpu
    set_session(tf.Session(config=config))
    from keras.models import load_model
    if not os.path.isfile(indir):
        wsis = glob.glob(os.path.join(indir, '*.tif'), recursive=False)
    else:
        wsis = [indir]
    os.makedirs(savedir, exist_ok=True)
    model = load_model(modelf)

    for wsi in tqdm(sorted(wsis)):
        basename = path_leaf(wsi).replace('.tif', '')
        #if basename!= 'tumor_110':
        #    continue
        print(basename)
        if jsondir:
            json_filepath = os.path.join(jsondir, basename + '.json')
            if not os.path.isfile(json_filepath):
                json_filepath = None
        else:
            json_filepath = None
        saveto = os.path.join(savedir, basename + '.joblib.pickle')
        saveto_original = os.path.join(savedir,
                                       basename + '.original.joblib.pickle')
        if os.path.isfile(saveto):
            # all done so continue
            continue
        all_samples = get_all_patches_from_slide(wsi, json_filepath, False,
                                                 patchsize)
        print(all_samples.head())
        if 'img_path' not in all_samples.columns:
            assert imgmaskdir is not None, 'Need to provide directory if img_path column is missing'
            tile_loc = all_samples.tile_loc.astype(str)
            tile_loc = tile_loc.str.replace(' ', '').str.replace(
                ')', '').str.replace('(', '')

            all_samples[['row', 'col']] = tile_loc.str.split(',', expand=True)
            all_samples['img_path'] = img_mask_dir + '/' + all_samples[[
                'uid', 'row', 'col'
            ]].apply(
                lambda x: '_'.join(x.values.tolist()),
                axis=1) + '.img.joblib.pickle'

            all_samples['mask_path'] = img_mask_dir + '/' + all_samples[[
                'uid', 'row', 'col'
            ]].apply(
                lambda x: '_'.join(x.values.tolist()),
                axis=1) + '.mask.joblib.pickle'
        if not os.path.isfile('/tmp/white.img.pickle'):
            white_img = np.ones(
                [patchsize, patchsize, 3], dtype=np.uint8) * 255
            joblib.dump(white_img, '/tmp/white.img.pickle')

        # Definitely not a tumor and hence all black
        if not os.path.isfile('/tmp/white.mask.pickle'):
            white_img_mask = np.ones(
                [patchsize, patchsize], dtype=np.uint8) * 0
            joblib.dump(white_img_mask, '/tmp/white.mask.pickle')

        all_samples.loc[all_samples.is_tissue == False,
                        'img_path'] = '/tmp/white.img.pickle'
        all_samples.loc[all_samples.is_tissue == False,
                        'mask_path'] = '/tmp/white.mask.pickle'

        for idx, row in all_samples.iterrows():
            f = row['img_path']
            if not os.path.isfile(f):
                row['savedir'] = imgmaskdir
                row['patch_size'] = patchsize
                row['index'] = idx
                save_images_and_mask(row)
        print(all_samples.head())
        slide = WSIReader(wsi, 40)
        n_cols = int(slide.dimensions[0] / patchsize)
        n_rows = int(slide.dimensions[1] / patchsize)
        n_samples = len(all_samples.index)
        assert n_rows * n_cols == len(
            all_samples.index), 'Some division error;'
        print('Total: {}'.format(len(all_samples.index)))
        predicted_thumbnails = list()

        for offset in tqdm(list(range(0, n_samples, batch_size))):
            batch_samples = all_samples.iloc[offset:offset + batch_size]
            X, _ = next(
                generate_tiles_fast(batch_samples, batch_size, shuffle=False))
            if batch_samples.is_tissue.nunique(
            ) == 1 and batch_samples.iloc[0].is_tissue == False:
                # all patches in this row do not have tissue, skip them all
                predicted_thumbnails.append(
                    np.zeros(batch_size, dtype=np.float32))
            else:
                # make predictions
                preds = predict_batch_from_model(X, model)
                predicted_thumbnails.append(preds.mean(axis=(1, 2)))
        predicted_thumbnails = np.asarray(predicted_thumbnails)
        try:
            output_thumbnail_preds = predicted_thumbnails.reshape(
                n_rows, n_cols)
            joblib.dump(output_thumbnail_preds, saveto)
        except:
            # Going to reshape
            flattened = predicted_thumbnails.ravel()[:n_rows * n_cols]
            print(
                'n_row = {} | n_col = {} | n_rowxn_col = {} | orig_shape = {} | flattened: {}'.
                format(n_rows, n_cols, n_rows * n_cols,
                       np.prod(predicted_thumbnails.shape), flattened.shape), )
            output_thumbnail_preds = flattened.reshape(n_rows, n_cols)
            joblib.dump(output_thumbnail_preds, saveto)
            slide.close()
            continue
        slide.close()


def generate_rows(samples, num_samples, batch_size=1):
    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]
            #is_tissue = batch_samples.is_tissue.tolist()
            #is_tumor = batch_samples.is_tumor.astype('int32').tolist()
            features = []
            labels = []
            #batch_samples = batch_samples.copy().drop(columns=['is_tissue', 'is_tumor'])
            for _, batch_sample in batch_samples.iterrows():
                row = batch_sample.values
                try:
                    label = int(batch_sample.is_tumor)
                except AttributeError:
                    # Should be normal
                    label = 0
                if batch_sample.is_tissue:
                    feature = pd.read_table(
                        os.path.join(
                            '/Z/personal-folders/interns/saket/github/pywsi',
                            batch_sample.segmented_tsv))

                    #feature = feature.drop(columns=['is_tumor', 'is_tissue'])
                    try:
                        feature = feature.loc[:, COLUMNS]
                        values = feature.loc[0].values
                        assert len(feature.columns) == 46
                    except KeyError:
                        # the segmentation returned empty columns!?
                        print(batch_sample.segmented_tsv)
                        #print(feature.columns)
                        #raise RuntimeError('Cannot parse the columns')
                        values = [0.0] * 46
                    features.append(values)
                else:
                    values = [0.0] * 46
                    features.append(values)
                labels.append(label)
            X_train = np.array(features, dtype=np.float32)
            y_train = np.array(labels)
            yield X_train, y_train


@cli.command(
    'heatmap-rf',
    context_settings=CONTEXT_SETTINGS,
    help='Create tumor probability heatmap using random forest model')
@click.option('--tif', help='Tif', required=True)
@click.option('--df', help='Root directory with all WSIs', required=True)
@click.option(
    '--modelf',
    help='Root directory with all jsons',
    default=
    '/Z/personal-folders/interns/saket/github/pywsi/models/random_forest_all_train.tf.model.meta'
)
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def create_tumor_map_rf_cmd(tif, df, modelf, patchsize, savedir):
    """Extract probability maps for a WSI
    """
    batch_size = 1
    all_samples = pd.read_table(df)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    from tensorflow.contrib.tensor_forest.python import tensor_forest
    from tensorflow.python.ops import resources
    os.makedirs(savedir, exist_ok=True)
    num_classes = 2
    num_features = 46
    num_trees = 100
    max_nodes = 10000
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(
        num_classes=num_classes,
        num_features=num_features,
        num_trees=num_trees,
        max_nodes=max_nodes).fill()

    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # Measure the accuracy
    infer_op, _, _ = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    saver = tf.train.import_meta_graph('{}'.format(modelf))
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(modelf)))

    slide = WSIReader(tif, 40)
    n_cols = int(slide.dimensions[0] / patchsize)
    n_rows = int(slide.dimensions[1] / patchsize)
    n_samples = len(all_samples.index)
    assert n_rows * n_cols == len(all_samples.index), 'Some division error;'
    print('Total: {}'.format(len(all_samples.index)))

    true_labels = []
    predicted_thumbnails = []
    #infer_op, accuracy_op, train_op, loss_op, X, Y = random_forest()
    for offset in tqdm(list(range(0, n_samples, batch_size))):
        batch_samples = all_samples.iloc[offset:offset + batch_size]
        X_test, true_label = next(generate_rows(batch_samples, batch_size))
        true_labels.append(true_label)
        if batch_samples.is_tissue.nunique(
        ) == 1 and batch_samples.iloc[0].is_tissue == False:
            # all patches in this row do not have tissue, skip them all
            predicted_thumbnails.append(0)
        else:
            preds = sess.run(infer_op, feed_dict={X: X_test})
            predicted_thumbnails.append(preds[0][1])
    predicted_thumbnails = np.asarray(predicted_thumbnails)

    basename = path_leaf(tif).replace('.tif', '')
    saveto = os.path.join(savedir, basename + '.joblib.pickle')
    saveto_original = os.path.join(savedir,
                                   basename + '.original.joblib.pickle')
    try:
        output_thumbnail_preds = predicted_thumbnails.reshape(n_rows, n_cols)
        joblib.dump(output_thumbnail_preds, saveto)
    except:
        # Going to reshape
        flattened = predicted_thumbnails.ravel()[:n_rows * n_cols]
        print(
            'n_row = {} | n_col = {} | n_rowxn_col = {} | orig_shape = {} | flattened: {}'.
            format(n_rows, n_cols, n_rows * n_cols,
                   np.prod(predicted_thumbnails.shape), flattened.shape), )
        output_thumbnail_preds = flattened.reshape(n_rows, n_cols)
        slide.close()


@cli.command(
    'add-patch-mask-col',
    context_settings=CONTEXT_SETTINGS,
    help='Add patch and mask column to a datamfrme')
@click.option('--df', help='Path to dataframe', required=True)
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--imgmaskdir',
    help='Directory where the patches and mask are stored',
    default=
    '/Z/personal-folders/interns/saket/github/pywsi/data/patch_img_and_mask/')
@click.option('--savedf', help='Save edited dataframe to', required=True)
@click.option(
    '--fast', help='Do not check of existense of images', is_flag=True)
def extract_patch_mask_cmd(df, patchsize, imgmaskdir, savedf, fast):
    """Extract tissue only patches from tumor WSIs.
    """
    img_mask_dir = imgmaskdir
    assert not os.path.isfile(savedf)
    df = pd.read_table(df)
    all_samples = df.copy()
    df['savedir'] = imgmaskdir
    df['patch_size'] = patchsize
    tile_loc = all_samples.tile_loc.astype(str)
    tile_loc = tile_loc.str.replace(' ', '').str.replace(')', '').str.replace(
        '(', '')

    all_samples[['row', 'col']] = tile_loc.str.split(',', expand=True)
    all_samples['img_path'] = img_mask_dir + '/' + all_samples[[
        'uid', 'row', 'col'
    ]].apply(
        lambda x: '_'.join(x.values.tolist()), axis=1) + '.img.joblib.pickle'

    all_samples['mask_path'] = img_mask_dir + '/' + all_samples[[
        'uid', 'row', 'col'
    ]].apply(
        lambda x: '_'.join(x.values.tolist()), axis=1) + '.mask.joblib.pickle'
    if not os.path.isfile('/tmp/white.img.pickle'):
        white_img = np.ones([patchsize, patchsize, 3], dtype=np.uint8) * 255
        joblib.dump(white_img, '/tmp/white.img.pickle')

    # Definitely not a tumor and hence all black
    if not os.path.isfile('/tmp/white.mask.pickle'):
        white_img_mask = np.ones([patchsize, patchsize], dtype=np.uint8) * 0
        joblib.dump(white_img_mask, '/tmp/white.mask.pickle')

    all_samples.loc[all_samples.is_tissue == False,
                    'img_path'] = '/tmp/white.img.pickle'
    all_samples.loc[all_samples.is_tissue == False,
                    'mask_path'] = '/tmp/white.mask.pickle'
    if not fast:
        for idx, row in all_samples.iterrows():
            f = row['img_path']
            if not os.path.isfile(f):
                row['savedir'] = imgmaskdir
                row['patch_size'] = patchsize
                row['index'] = idx
                save_images_and_mask(row)

    all_samples.to_csv(savedf, sep='\t', index=False, header=True)


@cli.command(
    'xmltojson',
    context_settings=CONTEXT_SETTINGS,
    help='Convert xmli coordinates to json')
@click.option(
    '--infile', help='Root directory with all tumor WSIs', required=True)
@click.option(
    '--savedir',
    help='Root directory to save extract images to',
    required=True)
def xmltojson_cmd(infile, savedir):
    """Convert ASAP xml files to json
    """
    xmltojson(infile, savedir)


@cli.command(
    'validate-mask-df',
    context_settings=CONTEXT_SETTINGS,
    help='Check if all files exist in img_path and mask_path columns')
@click.option('--df', help='Path to mask dataframe', required=True)
def validate_mask_cmd(df):
    df = pd.read_table(df)
    total = len(df.index)
    with tqdm(total=total) as pbar:
        for idx, row in df.iterrows():
            if not os.path.isfile(row['img_path']):
                print('Fixing {}'.format(row['img_path']))
                save_images_and_mask(row)
            pbar.update()


@cli.command(
    'validate-segmented-df',
    context_settings=CONTEXT_SETTINGS,
    help='Check if all files exist segmented_tsv path')
@click.option('--df', help='Path to segmented dataframe', required=True)
def validate_segmented_cmd(df):
    df = pd.read_table(df)
    total = len(df.index)
    with tqdm(total=total) as pbar:
        for idx, row in df.iterrows():
            if not os.path.isfile(row['segmented_tsv']):
                print('Fixing {}'.format(row['segmented_tsv']))
                process_segmentation_fixed(row)
            pbar.update()


def save_vahadane(filepaths):
    rgb_filepath, save_filepath = filepaths
    if '.pickle' not in rgb_filepath:
        rgb_patch = read_as_rgb(rgb_filepath)
    else:
        rgb_patch = joblib.load(rgb_filepath)

    vahadane_fit = VahadaneNormalization()
    try:
        vahadane_fit.fit(np.asarray(rgb_patch).astype(np.uint8))
    except:
        pass
    H_channel_v = vahadane_fit.get_hematoxylin_channel(rgb_patch)
    imsave(save_filepath, H_channel_v)


@cli.command(
    'convert-to-vahadane',
    context_settings=CONTEXT_SETTINGS,
    help='Check if all files exist segmented_tsv path')
@click.option('--df', help='Path to dataframe', required=True)
@click.option(
    '--savedir', help='Location to save segmented jpgs', required=True)
def convert_to_vahadane_cmd(df, savedir):
    aprun = ParallelExecutor(n_jobs=16)
    os.makedirs(savedir, exist_ok=True)
    df = pd.read_table(df)
    total = len(df.index)

    source_filepaths = df.img_path.tolist()
    dest_filepaths = [
        os.path.join(
            savedir,
            os.path.basename(x).replace('.png', '.jpg').replace(
                '.pickle', '.jpg')) for x in source_filepaths
    ]
    filepaths = list(zip(source_filepaths, dest_filepaths))
    with parallel_backend('threading', n_jobs=8):
        aprun(total=total)(delayed(save_vahadane)(f) for f in filepaths)
