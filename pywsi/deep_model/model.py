from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
from ..io.tiling import get_tiles, get_tiles_fast, get_all_patches_from_slide
from tqdm import tqdm
from multiprocessing import Pool


def get_model():
    """Load Keras model

    These parameters are probably not the most optimum.
    But, they seem to work (or I like to believe so.)

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    lambda_1 (Lambda)            (None, 256, 256, 3)       0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 128, 128, 100)     7600
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 64, 64, 100)       0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 200)       500200
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 16, 16, 200)       0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 16, 16, 300)       540300
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 16, 16, 400)       1080400
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 16, 400)       0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 400)       1440400
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 16, 16, 300)       1080300
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 16, 16, 300)       0
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 16, 16, 2)         602
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 256, 256, 2)       3846
    =================================================================
    Total params: 4,653,648
    Trainable params: 4,653,648
    Non-trainable params: 0
    _________________________________________________________________


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
    model.add(Convolution2D(2, (1, 1)))
    model.add(
        Conv2DTranspose(
            2, (31, 31),
            strides=(16, 16),
            activation='softmax',
            padding='same'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


def train(model, train_generator, n_train, n_epochs, batch_size,
          validation_generator, n_valid, checkpoint_prefix):
    """Train model

    Parameters
    ----------
    model: Keras.model
           Keras model as in load_model
    train_generator: generator
                     See generate_tiles method in pywsi.io.tiling
    n_train: int
             Total number of training samples
    batch_size: int
                Batch size

    validation_generator: generator
                          Generator for validation samples
    n_valid: int
             total number of validation samples
    checkpoint_prefix: string
                       Prefix for checkpoint files to be stored

    """
    os.makedirs(os.path.dirname(checkpoint_prefix))
    filepath = checkpoint_prefix + 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks_list = [checkpoint]
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    validation_steps = int(np.ceil(n_valid / batch_size))

    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=n_epochs,
        callbacks=callbacks_list)


def predict_on_patch(patch, model, patch_size=256):
    """Predict pixel level probabilites of being a tumor of a single patch.

    Parameters
    ----------
    patch: array_like
           RGB patch
    model: Keras model
           as obtained from get_model

    Returns
    -------
    prediction: array_like
                patch_size x patch_size x 1  per-pixel tumor probability
    """
    prediction = model.predict(patch.reshape(1, patch_size, patch_size, 3))
    prediction = prediction[:, :, :, 1].reshape(patch_size, patch_size)
    return prediction


def predict_on_batch(patches, model):
    """Predict pixel level probabilites of being a tumor of collection of patches.

    Parameters
    ----------
    patches: array_like
             A batch (list) of patches
    model: Keras model
           as obtained from get_model

    Returns
    -------
    prediction: array_like
                patch_size x patch_size x 1  per-pixel tumor probability
    """
    prediction = model.predict(patches)
    prediction = prediction[:, :, :, 1]
    return prediction


def process_batch(args):
    idx, model, batch_samples, batch_size, patch_size, img_mask_dir = args
    output_thumbnail_pred = None
    if batch_samples.is_tissue.nunique(
    ) == 1 and batch_samples.iloc[0].is_tissue == False:
        # all patches in this row do not have tissue, skip them all
        output_thumbnail_pred = np.zeros(batch_size, dtype=np.float32)

    else:
        # make predictions
        #X, _ = get_tiles(batch_samples)
        X, _ = get_tiles_fast(batch_samples, img_mask_dir=img_mask_dir)
        preds = predict_on_batch(X, model)
        output_thumbnail_pred = preds.mean(axis=(1, 2))
    return idx, output_thumbnail_pred


def slide_level_map(model,
                    slide_path,
                    batch_size=32,
                    patch_size=256,
                    json_filepath=None,
                    img_mask_dir=None):
    all_samples = get_all_patches_from_slide(slide_path, json_filepath, False,
                                             256)
    #all_samples = all_samples.sample(n=100)
    n_samples = len(all_samples.index)
    all_batch_samples = []
    for idx, offset in enumerate(list(range(0, n_samples, batch_size))):
        all_batch_samples.append(
            (idx, model, all_samples.iloc[offset:offset + batch_size],
             batch_size, patch_size, img_mask_dir))
    output_thumbnail_preds = []
    output_thumbnail_idx = []
    total = len(all_batch_samples)
    """
    with Pool(processes=32) as p:
        with tqdm(total=total) as pbar:
            for idx, result in p.imap_unordered(process_batch,
                                                all_batch_samples):
                output_thumbnail_preds.append(result)
                output_thumbnail_idx.append(idx)
                pbar.update()
    """
    for batch in tqdm(all_batch_samples):
        idx, result = process_batch(batch)
        output_thumbnail_preds.append(result)
        output_thumbnail_idx.append(idx)
    output_thumbnail_idx = np.array(output_thumbnail_idx)
    output_thumbnail_preds = np.array(output_thumbnail_preds)
    ##output_thumbnail_preds = output_thumbnail_preds[output_thumbnail_idx,:]
    return output_thumbnail_preds
