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

    print (tumor_wsis)
    last_used_x = None
    last_used_y = None
    stride = int(patchsize/2)
    for tumor_wsi in tumor_wsis:
        wsi = WSIReader(tumor_wsi, 40)
        uid = wsi.uid.replace('.tif', '')
        scale_factor = wsi.level0_mag/wsi.magnifications[level]
        normal_mask = np.load(
            os.path.join(maskdir, 'level_{}'.format(level),
                         uid + '_AnnotationNormalMask.npy'))
        tumor_mask = np.load(
            os.path.join(maskdir, 'level_{}'.format(level),
                         uid + '_AnnotationTumorMask.npy'))

        colored_patch= np.load(
            os.path.join(maskdir, 'level_{}'.format(level),
                         uid + '_AnnotationColored.npy'))
        subtracted_mask = tumor_mask * 1 - normal_mask * 1
        subtracted_mask[np.where(subtracted_mask < 0)] = 0
        x_ids, y_ids = np.where(subtracted_mask)
        for x_center, y_center in zip(x_ids, y_ids):
            out_file = '{}/level_{}/{}_{}_{}_{}.png'.format(savedir, level, uid, x_center,
                                                            y_center, patchsize)
            x_topleft = int(x_center - patchsize / 2)
            y_topleft = int(y_center - patchsize / 2)
            x_topright = x_topleft + patchsize
            y_bottomright = y_topleft + patchsize
            #print((x_topleft, x_topright, y_topleft, y_bottomright))
            mask = subtracted_mask[x_topleft:x_topright,
                                   y_topleft:y_bottomright]
            # Feed only complete cancer cells
            # Feed if more thatn 50% cells are cancerous!
            if np.sum(mask)>0.5*(patchsize*patchsize):
                if last_used_x is None:
                    last_used_x = x_center
                    last_used_y = y_center
                else:
                    diff_x = np.abs(x_center-last_used_x)
                    diff_y = np.abs(y_center-last_used_y)
                if diff_x>=stride and diff_y>=stride:
                    patch = colored_patch[x_topleft:x_topright,
                                          y_topleft:y_bottomright, :]
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    img = Image.fromarray(patch)
                    img.save(out_file)
                    last_used_x = x_center
                    last_used_y = y_center


if __name__ == '__main__':
    extract_tumor_patches_cmd()
