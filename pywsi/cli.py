# -*- coding: utf-8 -*-
"""Console script for pywsi."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import os
import json
import numpy as np
from six import iteritems

import click
from click_help_colors import HelpColorsGroup
import glob
from pywsi.io.operations import WSIReader, get_annotation_bounding_boxes, get_annotation_polygons,\
    poly2mask, translate_and_scale_polygon

from pywsi.morphology.patch_extractor import TissuePatch
from pywsi.morphology.mask import mpl_polygon_to_shapely_scaled, get_common_interior_polygons
from shapely.geometry import Polygon

from PIL import Image
click.disable_unicode_literals_warning = True
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
from tqdm import tqdm
import warnings


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
    default=256,
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
    default=256,
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
    default=256,
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
            filename, x0, y0 = line.split(',')
            filename = filename.lower()
            x0 = int(x0)
            y0 = int(y0)
            patches_to_extract[filename].append((x0, y0))

    for filename, coordinates in tqdm(patches_to_extract.items()):
        if 'normal' in filename:
            filepath = os.path.join(indir, 'normal', filename + '.tif')
        elif 'tumor' in filename:
            filepath = os.path.join(indir, 'tumor', filename + '.tif')
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
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--stride',
    default=128,
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
    default=256,
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


@cli.command(
    'extract-test-both-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Extract both normal and tumor patches from tissue masks')
@click.option(
    '--indir', help='Root directory with all test WSIs', required=True)
@click.option(
    '--patchsize',
    type=int,
    default=256,
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
def extract_test_both_cmd(indir, patchsize, stride, jsondir, level, savedir):
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
    #shape = (patchsize, patchsize)
    for wsi in wsis:
        last_used_x = None
        last_used_y = None
        wsi = WSIReader(wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        json_filepath = os.path.join(jsondir, uid + '.json')
        scale_factor = wsi.get_level_scale_factor(level)
        if not os.path.isfile(json_filepath):
            #warnings.warn('Skipping {} as no json found'.format(uid))
            continue
        json_parsed = json.load(open(json_filepath))
        tumor_patches = json_parsed['tumor']
        normal_patches = json_parsed['normal']
        polygons = {'tumor': [], 'normal': []}
        boxes = {'tumor': [], 'normal': []}
        for index, tumor_patch in enumerate(tumor_patches):
            polygon = np.array(tumor_patch['vertices'])
            polygon = translate_and_scale_polygon(
                polygon,
                x0=0,
                y0=0,
                scale_factor=scale_factor,
                edgecolor='tumor')
            polygon_s = mpl_polygon_to_shapely_scaled(
                polygon, x0=0, y0=0, scale_factor=scale_factor)
            #polygon_s = Polygon(polygon.get_xy())
            polygons['tumor'].append(polygon_s)
            boxes['tumor'].append(polygon_s.bounds)
        for index, normal_patch in enumerate(normal_patches):
            polygon = np.array(normal_patch['vertices'])
            polygon = translate_and_scale_polygon(
                polygon,
                x0=0,
                y0=0,
                scale_factor=scale_factor,
                edgecolor='normal')
            #polygon_s = Polygon(polygon.get_xy())
            polygon_s = mpl_polygon_to_shapely_scaled(
                polygon, x0=0, y0=0, scale_factor=scale_factor)
            polygons['normal'].append(polygon_s)
            boxes['normal'].append(polygon_s.bounds)

        polygons_to_exclude = {'tumor': [], 'normal': []}

        for polygon in polygons['tumor']:
            # Does this have any of the normal polygons inside it?
            polygons_to_exclude['tumor'].append(
                get_common_interior_polygons(polygon, polygons['normal']))

        for polygon in polygons['normal']:
            # Does this have any of the tumor polygons inside it?
            polygons_to_exclude['normal'].append(
                get_common_interior_polygons(polygon, polygons['tumor']))

        for polygon_key, polygon_value in iteritems(polygons):
            for index, polygon in enumerate(polygons[polygon_key]):
                minx, miny, maxx, maxy = boxes[polygon_key][index]
                width = int(maxx - minx)
                height = int(maxy - miny)
                shape = (width, height)
                # the polygon is already scaled so keep the scaling factpr 1
                polygon_translated = translate_and_scale_polygon(
                    polygon, minx, miny, 1)
                to_subtract_polygon_idx = polygons_to_exclude[polygon_key][
                    index]
                to_subtract_polygons = []
                for idx in to_subtract_polygon_idx:
                    to_subtract_polygon = polygons[polygon_key][idx]
                    to_subtract_polygons.append(
                        translate_and_scale_polygon(to_subtract_polygon, minx,
                                                    miny, 1))
                mask_full = poly2mask([polygon_translated], shape)
                current_polygon = Polygon(polygon_translated.get_xy())
                mask_interior = poly2mask(to_subtract_polygons, shape)
                mask_sub = mask_full - mask_interior
                mask_sub[np.where(mask_sub < 0)] = 0
                x_ids, y_ids = np.where(mask_sub)

                for x_topleft, y_topleft in zip(x_ids, y_ids):
                    out_file = os.path.join(
                        dirs[polygon_key], '{}_{}_{}_{}.png'.format(
                            uid, x_topleft, y_topleft, patchsize))
                    x_topright = x_topleft + patchsize
                    y_bottomright = y_topleft + patchsize
                    mask = mask_sub[x_topleft:x_topright, y_topleft:
                                    y_bottomright]
                    patch_polygon = Polygon(
                        [(x_topleft, y_topleft), (x_topright, y_topleft),
                         (x_topright, y_bottomright), (x_topright,
                                                       y_bottomright)])
                    if not patch_polygon.within(current_polygon):
                        #print(patch_polygon)
                        #print(current_polygon)
                        #raise RuntimeError
                        continue

                    if last_used_x is None:
                        last_used_x = x_topleft
                        last_used_y = y_topleft
                        diff_x = stride
                        diff_y = stride
                    else:
                        diff_x = np.abs(x_topleft - last_used_x)
                        diff_y = np.abs(y_topleft - last_used_y)
                    if diff_x >= stride and diff_y >= stride:
                        patch = wsi.get_patch_by_level(x_topleft, y_topleft,
                                                       level, patchsize)
                        os.makedirs(os.path.dirname(out_file), exist_ok=True)
                        img = Image.fromarray(patch)
                        img.save(out_file)
                        last_used_x = x_topleft
                        last_used_y = y_topleft
