# coding: utf-8

# In[1]:
import numpy as np
import joblib
import pandas as pd
import os
from pywsi.io.tiling import generate_tiles, get_all_patches_from_slide, generate_tiles_fast
patchsize = 32

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))
from sklearn.model_selection import StratifiedShuffleSplit


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop

from pywsi.io.tiling import generate_tiles, generate_tiles_fast

NUM_CLASSES = 2  # not_tumor, tumor
BATCH_SIZE = 32
N_EPOCHS = 50

# In[2]:
"""
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 256, 3)))
model.add(
    Convolution2D(
        100, (5, 5), strides=(2, 2), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(
    Convolution2D(
        200, (5, 5), strides=(2, 2), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(400, (3, 3), activation='elu', padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(400, (3, 3), activation='elu', padding='same'))
model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(
    2, (1, 1)))  # this is called upscore layer for some reason?
model.add(
    Conv2DTranspose(
        2, (31, 31), strides=(16, 16), activation='softmax', padding='same'))
#model = multi_gpu_model(model, gpus=2)

#model.compile(
#    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

opt = SGD(lr=1e-6, nesterov=True)
#opt = RMSProp(lr=0.01)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

# In[3]:
#model.load_weights('./rerun-allsamples-keras-improvement-01-0.70.hdf')

#model = load_model('./sgd-allsamples-keras-improvement-16-0.72.hdf')

"""

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 256, 3)))
model.add(Convolution2D(100, (5, 5), strides=(2, 2), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(200, (5, 5), strides=(2, 2), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(300, (3, 3), activation='elu', padding='same'))
model.add(Convolution2D(400, (3, 3), activation='elu',  padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(400, (3, 3), activation='elu',  padding='same'))
model.add(Convolution2D(300, (3, 3), activation='elu',  padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(2, (1, 1))) # this is called upscore layer for some reason?
model.add(Conv2DTranspose(2, (31, 31), strides=(16, 16), activation='softmax', padding='same'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


slide_path = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/tumor/tumor_110.tif'
json_path = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/lesion_annotations_json/tumor_110.json'
img_mask_dir = '/Z/personal-folders/interns/saket/github/pywsi/data/patch_img_and_mask/'

samples = get_all_patches_from_slide(slide_path=slide_path,
                                         filter_non_tissue=True,
                                         patch_size=256,
                                         json_filepath=json_path, )
if 'img_path' not in samples.columns:
    assert img_mask_dir is not None, 'Need to provide directory if img_path column is missing'
    tile_loc = samples.tile_loc.astype(str)
    tile_loc = tile_loc.str.replace(' ', '').str.replace(')',
                                                        '').str.replace(
                                                            '(', '')

    samples[['row', 'col']] = tile_loc.str.split(',', expand=True)
    samples['img_path'] = img_mask_dir + '/' + samples[[
        'uid', 'row', 'col'
    ]].apply(
        lambda x: '_'.join(x.values.tolist()),
        axis=1) + '.img.joblib.pickle'

    samples['mask_path'] = img_mask_dir + '/' + samples[[
        'uid', 'row', 'col'
    ]].apply(
        lambda x: '_'.join(x.values.tolist()),
        axis=1) + '.mask.joblib.pickle'
if not os.path.isfile('/tmp/white.img.pickle'):
    white_img = np.ones([patchsize, patchsize, 3], dtype=np.uint8) * 255
    joblib.dump(white_img, '/tmp/white.img.pickle')

# Definitely not a tumor and hence all black
if not os.path.isfile('/tmp/white.mask.pickle'):
    white_img_mask = np.ones([patchsize, patchsize], dtype=np.uint8) * 0
    joblib.dump(white_img_mask, '/tmp/white.mask.pickle')

samples.loc[samples.is_tissue == False,
                'img_path'] = '/tmp/white.img.pickle'
samples.loc[samples.is_tissue == False,
                'mask_path'] = '/tmp/white.mask.pickle'
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(samples, samples["is_tumor"]):
    train_samples = samples.loc[train_index]
    validation_samples = samples.loc[test_index]


def predict_from_model(patch, model):
    """Predict which pixels are tumor.

    input: patch: 256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """

    prediction = model.predict(patch.reshape(1, 256, 256, 3))
    prediction = prediction[:, :, :, 1].reshape(256, 256)
    return prediction


def predict_batch_from_model(patches, model):
    """Predict which pixels are tumor.

    input: patch: `batch_size`x256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    predictions = model.predict(patches)
    predictions = predictions[:, :, :, 1]
    return predictions


train_generator = generate_tiles_fast(
    train_samples.sample(32, random_state=42), 32, shuffle=True)
validation_generator = generate_tiles_fast(
    validation_samples.sample(32, random_state=42), 32, shuffle=True)

filepath = "newdropout-sgd-tumor_110-keras-improvement-{epoch:02d}-{val_acc:.2f}.hdf"

checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    train_generator,
    len(train_samples)//BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples)//BATCH_SIZE,
    epochs=N_EPOCHS,
    callbacks=callbacks_list)
