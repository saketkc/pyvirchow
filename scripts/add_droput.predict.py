# coding: utf-8
import pandas as pd
import os
# Just use 1 GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.visible_device_list = '0'
#set_session(tf.Session(config=config))
import click
from click_help_colors import HelpColorsGroup

from keras.models import load_model
from pyvirchow.io.tiling import generate_tiles, generate_tiles_fast
model = load_model('./newdropout-sgd-allsamples-keras-improvement-07-0.76.hdf')

predicted_thumbnails = slide_level_map(slide_path=slide_path, batch_size=32,
                                       img_mask_dir='/Z/personal-folders/interns/saket/github/pyvirchow/data/patch_img_and_mask/train_df',
                                       json_filepath=json_filepath, model=modeli
                                       )

