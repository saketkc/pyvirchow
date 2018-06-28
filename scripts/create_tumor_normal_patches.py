#!/usr/bin/env python
"""Command line interface to generate mask files for tumor/normal patches
from tumor WSIs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np

import click
import six
import pandas as pd
import glob
from pywsi.io.operations import WSIReader
from PIL import Image
click.disable_unicode_literals_warning = True
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def cli():
    """extract tumor patches"""
    pass


@click.command(
    'extract-tumor-patches',
    context_settings=CONTEXT_SETTINGS,
    help='Extract tumor patches from tumor WSIs')
@click.option('--indir', help='Root directory with all tumor WSIs', required=True)
@click.option('--maskdir', help='Root directory with all tumor WSIs', required=True)
@click.option('--level', type=int, help='Level at which to extract patches', required=True)
@click.option(
    '--patchsize',
    type=int,
    default=256,
    help='Patch size which to extract patches')
@click.option(
    '--stride', type=int, default=128, help='Stride to generate next patch')
@click.option('--savedir', help='Root directory to save extract images to', required=True)
def extract_tumor_patches_cmd(indir, maskdir, level, patchsize, stride,
                              savedir):
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
    for tumor_wsi in tumor_wsis:
        wsi = WSIReader(tumor_wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        normal_mask = np.load(
            os.path.join(maskdir, 'level{}'.format(level),
                         uid + '_AnnotationNormalMask.npy'))
        tumor_mask = np.load(
            os.path.join(maskdir, 'level{}'.format(level),
                         uid + '_AnnotationTumorMask.npy'))
        subtracted_mask = tumor_mask * 1 - normal_mask * 1
        subtracted_mask[np.where(subtracted_mask) < 0] = 0
        x_ids, y_ids = np.where(subtracted_mask)

        for x_center, y_center in zip(x_ids, y_ids):
            out_file = '{}/{}_{}_{}_{}.png'.format(savedir, uid, x_center,
                                                   y_center, patchsize)
            x_topleft = int(x_center - patchsize / 2)
            y_topleft = int(y_center - patchsize / 2)
            x_topright = x_topleft + patchsize
            y_bottomright = y_topleft + patchsize
            mask = subtracted_mask[np.arange(x_topleft, x_topright),
                                   np.arange(y_topleft, y_bottomright)]
            # Feed only complete cancer cells
            if np.all(mask):
                patch = wsi.get_patch_by_level(
                    x_topleft, y_topleft, level, patch_size=patchsize)
                img = Image.fromarray(patch)
                img.save(out_file)

if __name__ == '__main__':
    extract_tumor_patches_cmd()
