# -*- coding: utf-8 -*-
"""Console script for pywsi."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np

import click
from click_help_colors import HelpColorsGroup
import six
import pandas as pd
import glob
from pywsi.io.operations import WSIReader
from pywsi.morphology.patch_extractor import TissuePatch

from PIL import Image
click.disable_unicode_literals_warning = True
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
from tqdm import tqdm


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
    tumor_wsis = glob.glob(os.path.join(indir, 'tumor*.tif'), recursive=False)
    for tumor_wsi in tqdm(tumor_wsis):
        wsi = WSIReader(tumor_wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        json_filepath = os.path.join(jsondir, uid + '.json')
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
def extract_tumor_patches_cmd(indir, annmaskdir, tismaskdir, level, patchsize,
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
    tumor_wsis = glob.glob(os.path.join(indir, 'tumor*.tif'), recursive=False)

    # Assume that we want to generate these patches at level 0
    # So in order to ensure stride at a lower level
    # this needs to be discounted
    stride = int(patchsize / (2**level))
    for tumor_wsi in tqdm(tumor_wsis):
        last_used_x = None
        last_used_y = None
        wsi = WSIReader(tumor_wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        scale_factor = wsi.level0_mag / wsi.magnifications[level]
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
    stride = int(patchsize / (2**level))
    for wsi in tqdm(all_wsis):
        last_used_x = None
        last_used_y = None
        wsi = WSIReader(wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        scale_factor = wsi.level0_mag / wsi.magnifications[level]
        tissue_mask = np.load(
            os.path.join(tismaskdir, 'level_{}'.format(level),
                         uid + '_TissuePatch.npy'))
        if 'normal' in uid:
            # Just extract based on tissue patches
            x_ids, y_ids = np.where(tissue_mask)
            subtracted_mask = tissue_mask
            colored_patch = wsi.get_patch_by_level(0, 0, level)
        elif 'tumor' in uid:
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
