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
from pyvirchow.io.operations import WSIReader
from pyvirchow.morphology.patch_extractor import TissuePatch

from PIL import Image
click.disable_unicode_literals_warning = True
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def cli():
    """extract tumor patches"""
    pass


@click.command(
    'create-tissue-masks',
    context_settings=CONTEXT_SETTINGS,
    help='Extract tissue masks')
@click.option('--indir', help='Root directory with all tumor WSIs', required=True)
@click.option('--level', type=int, help='Level at which to extract patches', required=True)
@click.option('--savedir', help='Root directory to save extract images to', required=True)
def extract_tissue_masks_cmd(indir, level, savedir):
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
    for tumor_wsi in tumor_wsis:
        wsi = WSIReader(tumor_wsi, 40)
        tissue_patch = TissuePatch(wsi, level=level)
        uid = wsi.uid.replace('.tif', '')
        out_file = os.path.join(savedir, 'level_{}'.format(level), uid+'_TissuePatch.npy')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.save(out_file, tissue_patch.otsu_thresholded)

if __name__ == '__main__':
    extract_tissue_masks_cmd()
