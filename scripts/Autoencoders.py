# coding: utf-8
import numpy as np
import glob
from tqdm import tqdm
import os
from skimage.io import imread
import tensorflow as tf
import sklearn.preprocessing as prep
import joblib

def min_max_scale(X):
    preprocessor = prep.MinMaxScaler().fit(X)
    X_scaled = preprocessor.transform(X)
    return X_scaled


# ## import tensorflow as tf
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

# In[4]:

normal_patches_dir = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16_patches/normal_patches_test/level_0/'
tumor_patches_dir = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16_patches/tumor_patches_test/level_0/'

np.random.seed(42)
master_matrix = []
label_matrix = []
y = []
list_of_tumor_files = list(glob.glob('{}*.png'.format(tumor_patches_dir)))
list_of_normal_files = list(glob.glob('{}*.png'.format(normal_patches_dir)))

list_of_tumor_files = np.random.choice(list_of_tumor_files, 5000)
list_of_normal_files = np.random.choice(list_of_normal_files, 5000)

for f in tqdm(list_of_tumor_files):
    standardized = (imread(f) / 255.0).reshape(-1, 256 * 256 * 3)
    master_matrix.append(standardized)
    label_matrix.append('tumor')
    y.append(1)


for f in tqdm(list_of_normal_files):
    standardized = (imread(f) / 255.0).reshape(-1, 256 * 256 * 3)
    master_matrix.append(standardized)
    label_matrix.append('normal')
    y.append(0)


master_matrix = np.asarray(master_matrix)
joblib.dump(master_matrix, 'tumor_and_normal_200000_standardized_X.joblib.pickle')
y = np.array(y)
joblib.dump(y, 'tumor_and_normal_200000_standardized_y.joblib.pickle')

train_data = master_matrix
input_dim = train_data[0].shape


config = tf.ConfigProto(device_count={'GPU': 2})
config.gpu_options.allocator_type = 'BFC'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNELS = 3


class VAE(object):
    def __init__(self,
                 input_dim,
                 learning_rate=0.001,
                 n_latent=8,
                 batch_size=50):
        self.learning_rate = learning_rate
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.input_dim = input_dim

        self._build_network()
        self._create_loss_optimizer()

        init = tf.global_variables_initializer()
        #init = tf.initialize_all_variables()
        # Launch the session
        self.session = tf.InteractiveSession(config=config)
        self.session.run(init)
        self.saver = tf.train.Saver(tf.all_variables())

    def _build_network(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        dense1 = tf.layers.dense(
            activation=tf.nn.elu, inputs=self.x, units=512)
        dense2 = tf.layers.dense(
            activation=tf.nn.elu, inputs=dense1, units=512)
        dense3 = tf.layers.dense(
            activation=tf.nn.elu, inputs=dense2, units=512)
        dense4 = tf.layers.dense(
            activation=None, inputs=dense3, units=self.n_latent * 2)
        self.mu = dense4[:, :self.n_latent]
        self.sigma = tf.nn.softplus(dense4[:, self.n_latent:])
        eps = tf.random_normal(
            shape=tf.shape(self.sigma), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.mu + self.sigma * eps

        ddense1 = tf.layers.dense(
            activation=tf.nn.elu, inputs=self.z, units=512)
        ddense2 = tf.layers.dense(
            activation=tf.nn.elu, inputs=ddense1, units=512)
        ddense3 = tf.layers.dense(
            activation=tf.nn.elu, inputs=ddense2, units=512)

        self.reconstructed = tf.layers.dense(
            activation=tf.nn.sigmoid, inputs=ddense3, units=self.input_dim)

    def _create_loss_optimizer(self):
        epsilon = 1e-10
        reconstruction_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon + self.reconstructed) +
            (1 - self.x) * tf.log(epsilon + 1 - self.reconstructed),
            axis=1)

        self.reconstruction_loss = tf.reduce_mean(
            reconstruction_loss) / self.batch_size

        latent_loss = -0.5 * tf.reduce_sum(
            1 + tf.log(epsilon + self.sigma) - tf.square(self.mu) - tf.square(
                self.sigma),
            axis=1)
        latent_loss = tf.reduce_mean(latent_loss) / self.batch_size
        self.latent_loss = latent_loss
        self.cost = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
        # ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

    def fit_minibatch(self, batch):
        _, cost, reconstruction_loss, latent_loss = self.session.run(
            [
                self.optimizer, self.cost, self.reconstruction_loss,
                self.latent_loss
            ],
            feed_dict={self.x: batch})
        return cost, reconstruction_loss, latent_loss

    def reconstruct(self, x):
        return self.session.run([self.reconstructed], feed_dict={self.x: x})

    def decoder(self, z):
        return self.session.run([self.reconstructed], feed_dict={self.z: z})

    def encoder(self, x):
        return self.session.run([self.z], feed_dict={self.x: x})

    def save_model(self, checkpoint_path, epoch):
        self.saver.save(self.session, checkpoint_path, global_step=epoch)

    def load_model(self, checkpoint_path):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model: {}'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session,
                           checkpoint_path + '/' + ckpt.model_checkpoint_path)


# In[ ]:


def trainer(data,
            input_dim,
            learning_rate=1e-3,
            batch_size=100,
            num_epoch=50,
            n_latent=10,
            checkpoint_dir='/tmp/vae_checkpoint'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
    model = VAE(
        input_dim=input_dim,
        learning_rate=learning_rate,
        n_latent=n_latent,
        batch_size=batch_size)
    if os.path.isfile(checkpoint_path):
        model.load_model(checkpoint_path)
    total_losses = []
    reconstruction_losses = []
    latent_losses = []

    for epoch in range(num_epoch):
        #for (batch, labels) in iter.get_next():
        #    print(batch)
        for iter in range(num_sample // batch_size):
            batch = data[iter * batch_size:min((iter + 1) *
                                               batch_size, data.shape[0]), ]
            input_batch = batch[0]
            total_loss, reconstruction_loss, latent_loss = model.fit_minibatch(
                input_batch)
        latent_losses.append(latent_loss)
        reconstruction_losses.append(reconstruction_loss)
        total_losses.append(total_loss)

        if epoch % 5 == 0:
            print(
                '[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                    epoch, total_loss, reconstruction_loss, latent_loss))
            model.save_model(checkpoint_path, epoch)
            model.save_model(checkpoint_path+'-{}'.format(epoch), epoch)
            print("model saved to {}".format(checkpoint_path))

    print('Done!')
    return model, reconstruction_losses, latent_losses, total_losses


input_dims = input_dim[1]
num_sample = train_data.shape[0]
model, reconstruction_losses, latent_losses, total_losses = trainer(
    train_data,
    input_dims,
    learning_rate=1e-4,
    batch_size=32,
    num_epoch=1000,
    n_latent=10,
    checkpoint_dir=
    '/Z/personal-folders/interns/saket/vae_checkpoint_histoapath_2000')

