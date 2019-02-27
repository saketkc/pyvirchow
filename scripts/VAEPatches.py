
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA
import glob
from tqdm import tqdm
import os
import sklearn.preprocessing as prep
import pickle
import joblib
import tensorflow as tf
import pandas as pd
import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNELS = 3
BATCH_SIZE = 32

LEARNING_RATE = 1e-4
N_EPOCHS = 100
N_LATENT = 100
CHECKPOINT_DIR = '/Z/personal-folders/interns/saket/vae_patches_train_valid_nlatent100'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
INPUT_DIM = IMAGE_CHANNELS*IMAGE_WIDTH*IMAGE_HEIGHT


def min_max_scale(X):
    preprocessor = prep.MinMaxScaler().fit(X)
    X_scaled = preprocessor.transform(X)
    return X_scaled


# In[2]:


#config = tf.ConfigProto(
#    device_count = {'GPU': 0}
#)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = '1'

class VAE(object):
    def __init__(self,
                 input_dim,
                 learning_rate=0.01,
                 n_latent=8,
                 batch_size=50):
        self.learning_rate = learning_rate
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.input_dim = input_dim

        self._build_network()
        self._create_loss_optimizer()


        init = tf.global_variables_initializer()
        # Launch the session
        self.session = tf.Session(config=config)
        self.session.run(init)
        self.saver = tf.train.Saver(tf.global_variables())

    def close_session(self):
        self.session.close()

    def _build_network(self):
        self.x = tf.placeholder(tf.float32,
                                [None, self.input_dim])
        dense1 = tf.layers.dense(activation=tf.nn.elu,
                                 inputs=self.x,
                                 units=256)
        dense2 = tf.layers.dense(activation=tf.nn.elu,
                                 inputs=dense1,
                                 units=256)
        dense3 = tf.layers.dense(activation=tf.nn.elu,
                                 inputs=dense2,
                                 units=256)
        dense4 = tf.layers.dense(activation=None,
                                 inputs=dense3,
                                 units=self.n_latent * 2)
        self.mu = dense4[:, :self.n_latent]
        self.sigma = tf.nn.softplus(dense4[:, self.n_latent:])
        eps = tf.random_normal(shape=tf.shape(self.sigma),
                               mean=0,
                               stddev=1,
                               dtype=tf.float32)
        self.z = self.mu + self.sigma * eps

        ddense1 = tf.layers.dense(activation=tf.nn.elu,
                                  inputs=self.z,
                                  units=256)
        ddense2 = tf.layers.dense(activation=tf.nn.elu,
                                  inputs=ddense1,
                                  units=256)
        ddense3 = tf.layers.dense(activation=tf.nn.elu,
                                  inputs=ddense2,
                                  units=256)

        self.reconstructed = tf.layers.dense(activation=tf.nn.sigmoid,
                                             inputs=ddense3,
                                             units=self.input_dim)

    def _create_loss_optimizer(self):
        epsilon = 1e-10
        reconstruction_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.reconstructed) + (1-self.x) * tf.log(epsilon+1-self.reconstructed),
            axis=1
        )

        self.reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(epsilon+self.sigma) - tf.square(self.mu) - tf.square(self.sigma),
                                           axis=1)
        latent_loss = tf.reduce_mean(latent_loss)
        self.latent_loss = latent_loss
        self.cost = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
        # ADAM optimizer
        self.optimizer =             tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


    def fit_minibatch(self, batch):
        _, cost, reconstruction_loss, latent_loss = self.session.run([self.optimizer,
                                                                      self.cost,
                                                                      self.reconstruction_loss,
                                                                      self.latent_loss],
                                                                     feed_dict = {self.x: batch})
        return  cost, reconstruction_loss, latent_loss

    def reconstruct(self, x):
        return self.session.run([self.reconstructed], feed_dict={self.x: x})

    def decoder(self, z):
        return self.session.run([self.reconstructed], feed_dict={self.z: z})

    def encoder(self, x):
        return self.session.run([self.z], feed_dict={self.x: x})

    def save_model(self, checkpoint_path, epoch):
        self.saver.save(self.session, checkpoint_path, global_step = epoch)

    def load_model(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir, latest_filename='checkpoint')
        print('loading model: {}'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session, ckpt.model_checkpoint_path)


# In[3]:


train_df_file = '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/train_df_with_mask.tsv'
valid_df_file = '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/validate_df_with_mask.tsv'
train_df = pd.read_table(train_df_file)
train_df.columns#()


# In[4]:


train_df_file = '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/train_df_with_mask.tsv'
valid_df_file = '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/validate_df_with_mask.tsv'
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

train_samples.to_csv('./train_samples_subsampled.tsv', sep='\t', index=False, header=True)
validation_samples.to_csv('./validation_samples_subsampled.tsv', sep='\t', index=False, header=True)


def preprocess(image):
    return image/255.0 - 0.5

def _read_py_function(label, filename):
    image_decoded = joblib.load(filename)
    image_decoded = preprocess(image_decoded)
    #image_decoded = min_max_scale(image_decoded)
    #print(label)
    #print(image_decoded)
    return np.int32(eval(label)), image_decoded.astype(np.float32)

def _resize_function(label, image_decoded):
    image_resized = tf.reshape(image_decoded, (-1, INPUT_DIM))
    image_resized = tf.cast(
        image_resized,
        tf.float32)
    return tf.cast(label, tf.int32), image_resized


def make_dataset(df):
    record_defaults = [[''], ['']]
    select_cols = [1, 7]
    dataset = tf.contrib.data.CsvDataset(df,
                                         record_defaults,
                                         header=True,
                                         field_delim='\t',
                                         select_cols=select_cols)
    #training_dataset = training_dataset.map(parser,
    #num_parallel_calls=BATCH_SIZE)
    dataset = dataset.map( lambda is_tumor, img_path: tuple(tf.py_func(_read_py_function,
                                                                       [is_tumor, img_path],
                                                                       [np.int32, np.float32])))
    dataset = dataset.map(_resize_function)
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


# In[5]:


training_dataset = make_dataset('./train_samples_subsampled.tsv')
validation_dataset = make_dataset('./validation_samples_subsampled.tsv')
training_iterator = training_dataset.make_one_shot_iterator()

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
training_init_op = iterator.make_initializer(training_dataset)
#validation_init_op = iterator.make_initializer(validation_dataset)


# In[6]:


model = VAE(input_dim=INPUT_DIM,
            learning_rate=LEARNING_RATE,
            n_latent=N_LATENT,
            batch_size=BATCH_SIZE)
total_losses = []
reconstruction_losses = []
latent_losses = []
sess = model.session
training_next_batch = iterator.get_next()

for epoch in range(N_EPOCHS):
    sess.run(training_init_op)
    while True:
        try:
            training_label_batch, training_image_batch = sess.run(training_next_batch)
            #print(training_image_batch)
            #print(training_label_batch)
        except tf.errors.OutOfRangeError:
            break
        input_batch = training_image_batch
        #input_batch = np.reshape(input_batch, (-1, ))
        input_batch = np.asarray(input_batch, dtype=np.float32).reshape(-1, 256*256*3)
        total_loss, reconstruction_loss, latent_loss = model.fit_minibatch(input_batch)
        latent_losses.append(latent_loss)
        reconstruction_losses.append(reconstruction_loss)
        total_losses.append(total_loss)
        total_losses_path = os.path.join(CHECKPOINT_DIR, 'total_losses.pickle')
        latent_losses_path = os.path.join(CHECKPOINT_DIR, 'latent_losses.pickle')
        reconstruction_losses_path = os.path.join(CHECKPOINT_DIR, 'reconstruction_losses.pickle')
        joblib.dump(total_losses, total_losses_path)
        joblib.dump(latent_losses, latent_losses_path)
        joblib.dump(reconstruction_losses, reconstruction_losses_path)

    if epoch % 5 == 0:
        print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
            epoch, total_loss, reconstruction_loss, latent_loss))
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model.ckpt')
        model.save_model(checkpoint_path, epoch)
        print ("model saved to {}".format(checkpoint_path))

print('Done!')
#return model, reconstruction_losses, lat


# In[ ]:


training_image_batch

