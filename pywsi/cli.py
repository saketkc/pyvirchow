# -*- coding: utf-8 -*-
"""Console script for pywsi."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pywsi.io.operations import get_annotation_bounding_boxes
from pywsi.io.operations import get_annotation_polygons
from pywsi.io.operations import path_leaf
from pywsi.io.operations import read_as_rgb
from pywsi.io.operations import WSIReader
from pywsi.io.tiling import get_all_patches_from_slide

from pywsi.morphology.patch_extractor import TissuePatch
from pywsi.morphology.mask import get_common_interior_polygons
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from pywsi.segmentation import label_nuclei, summarize_region_properties

from collections import defaultdict
import os
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
warnings.filterwarnings('ignore')


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
    print(indir)
    list_of_pngs = list(glob.glob(indir + '/*.png'))
    print(os.path.join(indir, '/{}*.png'))
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
        df = df.sort_values(by=['uid'])

    df.to_csv(
        os.path.join(savedir, 'master_df.tsv'),
        sep='\t',
        index=False,
        header=True)
