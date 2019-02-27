# coding: utf-8

# In[1]:
import numpy as np
import joblib
import pandas as pd
import os
from pyvirchow.io.tiling import generate_tiles, get_all_patches_from_slide, generate_tiles_fast
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

from pyvirchow.io.tiling import generate_tiles, generate_tiles_fast

NUM_CLASSES = 2  # not_tumor, tumor
BATCH_SIZE = 32
N_EPOCHS = 50

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

train_samples = pd.read_table(
    '/Z/personal-folders/interns/saket/github/pyvirchow/data/patch_df/train_df_with_mask.tsv'
)
validation_samples = pd.read_table(
    '/Z/personal-folders/interns/saket/github/pyvirchow/data/patch_df/validate_df_with_mask.tsv'
)

if not os.path.isfile('/tmp/white.img.pickle'):
    white_img = np.ones([patchsize, patchsize, 3], dtype=np.uint8) * 255
    joblib.dump(white_img, '/tmp/white.img.pickle')

# Definitely not a tumor and hence all black
if not os.path.isfile('/tmp/white.mask.pickle'):
    white_img_mask = np.ones([patchsize, patchsize], dtype=np.uint8) * 0
    joblib.dump(white_img_mask, '/tmp/white.mask.pickle')

train_samples.loc[train_samples.is_tissue == False,
                'img_path'] = '/tmp/white.img.pickle'
train_samples.loc[train_samples.is_tissue == False,
                'mask_path'] = '/tmp/white.mask.pickle'
validation_samples.loc[validation_samples.is_tissue == False,
                'img_path'] = '/tmp/white.img.pickle'
validation_samples.loc[validation_samples.is_tissue == False,
                'mask_path'] = '/tmp/white.mask.pickle'

split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_index, test_index in split.split(train_samples, train_samples["is_tumor"]):
    train_samples = train_samples.loc[train_index]
#    validation_samples = samples.loc[test_index]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_index, test_index in split.split(validation_samples, validation_samples["is_tumor"]):
    validation_samples = validation_samples.loc[train_index]


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

filepath = "8conv-adam-keras-improvement-{epoch:02d}-{val_acc:.2f}.hdf"

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
