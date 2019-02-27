# coding: utf-8

# In[1]:
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from keras_tqdm import TQDMCallback

# Just use 1 GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))
from sklearn.metrics import confusion_matrix

import joblib
from keras.models import load_model

from pywsi.io.tiling import generate_tiles_fast

NUM_CLASSES = 2  # not_tumor, tumor
BATCH_SIZE = 32
N_EPOCHS = 50

# In[2]:

model = load_model('./newdropout-sgd-allsamples-keras-improvement-15-0.76.hdf')

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


#train_samples = train_samples.sample(frac=0.5, random_state=42)
train_samples = pd.concat([train_samples_tumor, train_samples_normal]).sample(frac=1, random_state=42)
#
#validation_samples = validation_samples.sample(frac=0.5, random_state=43)
validation_samples = pd.concat([validation_samples_tumor, validation_samples_normal]).sample(frac=1, random_state=43)

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
    train_samples, 32, shuffle=False)
validation_generator = generate_tiles_fast(
    validation_samples, 32, shuffle=False)
train_steps = len(train_samples)//BATCH_SIZE
validation_steps = len(validation_samples)//BATCH_SIZE

#train_predictions = model.predict_generator(train_generator, steps= len(train_samples)//BATCH_SIZE)

#validation_predictions = model.predict_generator(validation_generator,
#                                                 steps= len(validation_samples)//BATCH_SIZE,
#                                                 callbacks=[TQDMCallback()])
#print(validation_predictions.shape)


#train_predictions = np.argmax(predictions, axis=-1) #multiple categories
#joblib.dump(validation_predictions, './newdropout-sgd-allsamples-keras-improvement-13-0.76-validation-prediction.joblib.pickle')
def predict_batch_from_model(patches, model):
    """Predict which pixels are tumor.

    input: patch: `batch_size`x256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    predictions = model.predict(patches)
    predictions = predictions[:, :, :, 1]
    return predictions

train_true = []
train_pred = []

confusion_mtx = np.zeros((2, 2))

def is_patch_tumor(t):
    # If there are 10 pixels with 1s,
    # it should be just tumor
    return int(np.sum(t) >= 10)

## These accuracies are pixel level which to be honest
## is to much to ask for

val_true = []
val_pred = []
confusion_mtx = np.zeros((2, 2))

for i in tqdm(range(int(validation_steps))):
    X, y  = next(validation_generator)
    preds = predict_batch_from_model(X, model)

    y_true = y[:, :, :, 1].ravel()
    y_pred = np.uint8(preds > 0.5).ravel()

    y_true1 = y[:, :, :, 1]
    y_pred1 = preds[:, :, :]
    shape = y.shape

    y_true1_summary = y_true1.reshape(shape[0], shape[1]*shape[2])
    y_pred1_summary = y_pred1.reshape(shape[0], shape[1]*shape[2])

    y_true_label = [is_patch_tumor(x) for x in y_true1_summary]
    y_pred_label = [is_patch_tumor(x) for x in y_pred1_summary]

    val_true += y_true_label
    val_pred += y_pred_label
    confusion_mtx += confusion_matrix(y_true, y_pred, labels=[0, 1])
joblib.dump(confusion_matrix, 'dropout_sgd_val_confusion_matrix.joblib.pickle')
joblib.dump(val_true, 'dropout_sgd_val_true.joblib.pickle')
joblib.dump(val_pred, 'dropout_sgd_val_pred.joblib.pickle')
tn = confusion_mtx[0, 0]
fp = confusion_mtx[0, 1]
fn = confusion_mtx[1, 0]
tp = confusion_mtx[1, 1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Val Accuracy: %.2f" % accuracy)
print("Val Recall: %.2f" % recall)
print("Val Precision: %.2f" % precision)
print("Val F1 Score: %.2f" % f1_score)

confusion_mtx = confusion_matrix(val_true, val_pred, labels=[0, 1])
tn = confusion_mtx[0, 0]
fp = confusion_mtx[0, 1]
fn = confusion_mtx[1, 0]
tp = confusion_mtx[1, 1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Val (patch) Accuracy: %.2f" % accuracy)
print("Val (patch) Recall: %.2f" % recall)
print("Val (patch) Precision: %.2f" % precision)
print("Val (patch) F1 Score: %.2f" % f1_score)


for i in tqdm(range(int(train_steps))):
    X, y  = next(train_generator)
    preds = predict_batch_from_model(X, model)

    y_true = y[:, :, :, 1].ravel()
    y_pred = np.uint8(preds > 0.5).ravel()


    y_true1 = y[:, :, :, 1]
    y_pred1 = preds[:, :, :]
    shape = y.shape

    y_true1_summary = y_true1.reshape(shape[0], shape[1]*shape[2])
    y_pred1_summary = y_pred1.reshape(shape[0], shape[1]*shape[2])

    y_true_label = [is_patch_tumor(x) for x in y_true1_summary]
    y_pred_label = [is_patch_tumor(x) for x in y_pred1_summary]

    train_true += y_true_label
    train_pred += y_pred_label

    confusion_mtx += confusion_matrix(y_true, y_pred, labels=[0, 1])
joblib.dump(confusion_matrix, 'dropout_sgd_train_confusion_matrix.joblib.pickle')
joblib.dump(train_true, 'dropout_sgd_train_true.joblib.pickle')
joblib.dump(train_pred, 'dropout_sgd_train_pred.joblib.pickle')

tn = confusion_mtx[0, 0]
fp = confusion_mtx[0, 1]
fn = confusion_mtx[1, 0]
tp = confusion_mtx[1, 1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Train Accuracy: %.2f" % accuracy)
print("Train Recall: %.2f" % recall)
print("Train Precision: %.2f" % precision)
print("Train F1 Score: %.2f" % f1_score)

confusion_mtx = confusion_matrix(train_true, train_pred, labels=[0, 1])
tn = confusion_mtx[0, 0]
fp = confusion_mtx[0, 1]
fn = confusion_mtx[1, 0]
tp = confusion_mtx[1, 1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Train (patch) Accuracy: %.2f" % accuracy)
print("Train (patch) Recall: %.2f" % recall)
print("Train (patch) Precision: %.2f" % precision)
print("Train (patch) F1 Score: %.2f" % f1_score)


