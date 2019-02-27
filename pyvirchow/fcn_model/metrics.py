import keras.backend as K
import tensorflow as tf


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(
        tf.to_float(
            legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
        )
    ) / K.sum(tf.to_float(legal_labels))
