import pandas as pd
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.visible_device_list = '0,1'
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop
from keras.optimizers import Adam, SGD
from keras.layers import *

from pyvirchow.io.tiling import generate_tiles, generate_tiles_fast

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(
        tf.to_int32(K.flatten(y_true)),
        K.int_shape(y_pred)[-1] + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def resize_images_bilinear(X,
                           height_factor=1,
                           width_factor=1,
                           target_height=None,
                           target_width=None,
                           data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(
                np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor,
                         original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(
                np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor,
                         original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(Layer):
    def __init__(self,
                 size=(1, 1),
                 target_size=None,
                 data_format='default',
                 **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'
                               }, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] *
                        input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3]
                         if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0], input_shape[1], width, height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] *
                        input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2]
                         if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0], width, height, input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(
                x,
                target_height=self.target_size[0],
                target_width=self.target_size[1],
                data_format=self.data_format)
        else:
            return resize_images_bilinear(
                x,
                height_factor=self.size[0],
                width_factor=self.size[1],
                data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

NUM_CLASSES = 2  # not_tumor, tumor
BATCH_SIZE = 32
N_EPOCHS = 50

# In[2]:

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 256, 3)))
model.add(
    Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv1'))
model.add(
    Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(
    Conv2D(
        128, (3, 3), activation='elu', padding='same', name='block2_conv1'))
model.add(
    Conv2D(
        128, (3, 3), activation='elu', padding='same', name='block2_conv2'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(
    Conv2D(
        256,
        (3, 3),
        activation='elu',
        padding='same',
        name='block3_conv1',
    ))
model.add(
    Conv2D(
        256,
        (3, 3),
        activation='elu',
        padding='same',
        name='block3_conv2',
    ))
model.add(
    Conv2D(
        256,
        (3, 3),
        activation='elu',
        padding='same',
        name='block3_conv3',
    ))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(
    Conv2D(
        512,
        (3, 3),
        activation='elu',
        padding='same',
        name='block4_conv1',
    ))
model.add(
    Conv2D(
        512,
        (3, 3),
        activation='elu',
        padding='same',
        name='block4_conv2',
    ))
model.add(
    Conv2D(
        512,
        (3, 3),
        activation='elu',
        padding='same',
        name='block4_conv3',
    ))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
model.add(Dropout(0.5))

# Block 5
model.add(
    Conv2D(
        512,
        (3, 3),
        activation='elu',
        padding='same',
        name='block5_conv1',
    ))
model.add(
    Conv2D(
        512,
        (3, 3),
        activation='elu',
        padding='same',
        name='block5_conv2',
    ))
model.add(
    Conv2D(
        512,
        (3, 3),
        activation='elu',
        padding='same',
        name='block5_conv3',
    ))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
model.add(Dropout(0.5))

# Convolutional layers transfered from fully-connected layers
model.add(
    Conv2D(
        4096,
        (7, 7),
        activation='elu',
        padding='same',
        name='fc1',
    ))
model.add(Dropout(0.5))
model.add(
    Conv2D(
        4096,
        (1, 1),
        activation='elu',
        padding='same',
        name='fc2',
    ))
model.add(Dropout(0.5))
#classifying layer
model.add(
    Conv2D(
        NUM_CLASSES,
        (1, 1),
        kernel_initializer='he_normal',
        activation='linear',
        padding='valid',
        strides=(1, 1),
    ))

model.add(
    Conv2DTranspose(
        2, (64, 64), strides=(32, 32), activation='softmax', padding='same'))

#model.add(BilinearUpSampling2D(target_size=(256, 256,2)))
#    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING  )(o)
#        o = (Activation('softmax'))(o)

opt = Adam(lr=1e-6)  # nesterov=True)
opt = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
print(model.summary())
#model.load_weights('./allsamples-keras-improvement-03-0.62.hdf')
# In[3]:
#model = load_model('./fast-allsamples-keras-improvement-01-0.37.hdf')

train_samples = pd.read_table(
    '/home/saketkc/github/pyvirchow/data/df/train_df_with_mask.tsv')
validation_samples = pd.read_table(
    '/home/saketkc/github/pyvirchow/data/df/validate_df_with_mask.tsv')

# In[4]:

# Sample only half the points
train_samples_tumor = train_samples[train_samples.is_tumor == True].sample(
    frac=0.45, random_state=42)
train_samples_normal = train_samples[train_samples.is_tumor == False].sample(
    frac=0.45, random_state=43)

validation_samples_tumor = validation_samples[validation_samples.is_tumor ==
                                              True].sample(
                                                  frac=0.45, random_state=42)
validation_samples_normal = validation_samples[validation_samples.is_tumor ==
                                               False].sample(
                                                   frac=0.45, random_state=43)

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

filepath = "deeper-allsamples-keras-improvement-{epoch:02d}-{val_acc:.2f}.hdf"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    train_generator,
    len(train_samples) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples) // BATCH_SIZE,
    epochs=N_EPOCHS,
    callbacks=callbacks_list)
