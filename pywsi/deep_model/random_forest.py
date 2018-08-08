import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

BUFFER_SIZE = 1000
NUM_FEATURES = 46
NUM_COLUMNS = NUM_FEATURES + 1


def _parse_csv(rows_string_tensor):
    """Takes the string input tensor and returns tuple of (features, labels)."""
    # First dim is the label.
    columns = tf.decode_csv(
        rows_string_tensor,
        record_defaults=[[0.0]] * NUM_COLUMNS,
        field_delim='\t')
    label = columns[0]
    return tf.cast(tf.stack(columns[1:]), tf.float32), tf.cast(label, tf.int32)


def input_fn(file_names, batch_size):
    """The input_fn."""
    dataset = tf.data.TextLineDataset(file_names).skip(1)
    # Skip the first line (which does not have data).
    dataset = dataset.map(_parse_csv)
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    next_batch = iterator.get_next()
    init_op = iterator.make_initializer(dataset)
    return init_op, next_batch


def random_forest(num_classes=2,
                  num_features=46,
                  num_trees=100,
                  max_nodes=10000):
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(
        num_classes=num_classes,
        num_features=num_features,
        num_trees=num_trees,
        max_nodes=max_nodes).fill()

    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # Measure the accuracy
    infer_op, _, _ = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_vars = tf.group(
        tf.global_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()))

    sess = tf.Session()
    sess.run(init_vars)
    return sess


def train_rf(sess, train_df_file, batch_size=1024, num_steps=100):
    training_init_op, training_next_batch = input_fn([train_df_file],
                                                     batch_size)
    previous_loss = None
    loss = None
    for epoch in range(num_steps):
        sess.run(training_init_op)
        while True:
            try:
                training_features_batch, training_label_batch = sess.run(
                    training_next_batch)
            except tf.errors.OutOfRangeError:
                break
            previous_loss = loss
            _, loss = sess.run(
                [train_op, loss_op],
                feed_dict={
                    X: training_features_batch,
                    Y: training_label_batch
                })
        acc = sess.run(
            accuracy_op,
            feed_dict={
                X: training_features_batch,
                Y: training_label_batch
            })
        print('Step %i, Loss: %f, Acc: %f' % (epoch, loss, acc))
        if loss == previous_loss:
            break


def accuracy_rf(sess, df_file, batch_size=1):
    accuracies = []
    init_op, next_batch = input_fn([df_file], batch_size)
    sess.run(init_op)
    while True:
        try:
            features_batch, label_batch = sess.run(next_batch)
        except tf.errors.OutOfRangeError:
            break
        acc = sess.run(
            accuracy_op, feed_dict={
                X: features_batch,
                Y: label_batch
            })
        accuracies.append(acc)
    return accuracies


def probability_rf(sess, df_file, batch_size=1):
    probabilities = []
    init_op, next_batch = input_fn([df_file], batch_size)
    sess.run(init_op)
    while True:
        try:
            features_batch, label_batch = sess.run(next_batch)
        except tf.errors.OutOfRangeError:
            break
        prob = sess.run(infer_op, feed_dict={X: features_batch})
        probabilities.append(prob[0][1])
    return probabilities
