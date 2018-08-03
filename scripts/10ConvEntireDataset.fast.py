# coding: utf-8

# In[1]:

import pandas as pd

import os
# Just use 1 GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model

from pywsi.io.tiling import generate_tiles_fast

NUM_CLASSES = 2  # not_tumor, tumor
BATCH_SIZE = 32
N_EPOCHS = 20

# In[2]:

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

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.load_weights('./allsamples-keras-improvement-03-0.62.hdf')
# In[3]:

train_samples = pd.read_table(
    '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/train_df_with_mask.tsv'
)
validation_samples = pd.read_table(
    '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/validate_df_with_mask.tsv'
)

# In[4]:

# Sample only half the points
train_samples_tumor = train_samples[train_samples.is_tumor==True].sample(frac=0.45, random_state=42)
train_samples_normal = train_samples[train_samples.is_tumor==False].sample(frac=0.45, random_state=43)

validation_samples_tumor = validation_samples[validation_samples.is_tumor==True].sample(frac=0.45, random_state=42)
validation_samples_normal = validation_samples[validation_samples.is_tumor==False].sample(frac=0.45, random_state=43)


#train_samples = pd.concat([train_samples_tumor, train_samples_normal]).sample(frac=1, random_state=42)
#
#validation_samples = pd.concat([validation_samples_tumor, validation_samples_normal]).sample(frac=1, random_state=42)


#train_samples = train_samples.sample(frac=0.5, random_state=42)
#validation_samples = validation_samples.sample(frac=0.5, random_state=43)
# Let's try on tumor_076 samples
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

filepath = "fast-allsamples-keras-improvement-{epoch:02d}-{val_acc:.2f}.hdf"
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
