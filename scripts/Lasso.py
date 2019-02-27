import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import glob
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import fitsne
from sklearn.model_selection import cross_val_score

from skimage.color import rgb2gray

from pyvirchow.io import WSIReader
from pyvirchow.io.operations import read_as_rgb

from pyvirchow.segmentation import poisson_deconvolve, perform_binary_cut, max_clustering
from pyvirchow.segmentation import collapse_labels, collapse_small_area, laplace_of_gaussian
from pyvirchow.segmentation import gmm_thresholding, label_nuclei, extract_features, summarize_region_properties

from pyvirchow.normalization import MacenkoNormalization
from pyvirchow.normalization import ReinhardNormalization
from pyvirchow.normalization import VahadaneNormalization
from pyvirchow.normalization import XuNormalization

from sklearn.decomposition import PCA, FastICA
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=2)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from tpot import TPOTClassifier
import pickle
from multiprocessing import Pool

scaler = StandardScaler()

# In[ ]:

normal_patches_dir = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16_patches/normal_patches_test/level_0/'
tumor_patches_dir = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16_patches/tumor_patches_test/level_0/'

np.random.seed(42)
list_of_tumor_files = list(glob.glob('{}*.png'.format(tumor_patches_dir)))
list_of_normal_files = list(glob.glob('{}*.png'.format(normal_patches_dir)))


def draw_nuclei(patch,
                local_max_search_radius=3,
                min_radius=5,
                max_radius=15,
                min_nucleus_area=100):
    patch = read_as_rgb(patch)
    label_nuclei(
        patch,
        local_max_search_radius=local_max_search_radius,
        min_radius=min_radius,
        max_radius=max_radius,
        min_nucleus_area=min_nucleus_area)


features_df = []
labels = []


def process_sample(sample):
    patch = read_as_rgb(sample)
    region_properties, _ = label_nuclei(patch, draw=False)
    summary = summarize_region_properties(region_properties, patch)
    return summary


with tqdm(total=len(list_of_tumor_files)) as pbar:
    with Pool(processes=32) as p:
        for i, summary in enumerate(
                p.imap_unordered(process_sample, list_of_tumor_files)):
            pbar.update()
            if summary is None:
                sample = list_of_tumor_files[i]
                print('Nothing found for {}'.format(sample))
                continue
            else:
                labels.append('tumor')
    features_df.append(summary)

with tqdm(total=len(list_of_normal_files)) as pbar:
    with Pool(processes=32) as p:
        for i, summary in enumerate(
                p.imap_unordered(process_sample, list_of_normal_files)):
            pbar.update()
            if summary is None:
                sample = list_of_normal_files[i]
                print('Nothing found for {}'.format(sample))
                continue
            else:
                labels.append('normal')
    features_df.append(summary)

y = np.array([1 if label == 'normal' else 0 for label in labels])

f = pd.DataFrame(features_df)
X = f.values

X_scaled = scaler.fit(X).transform(X)

embedding = umap.UMAP(
    n_neighbors=20, min_dist=0.3, metric='correlation').fit_transform(X_scaled)
fig = plt.figure(figsize=(10, 10))
colors = ['navy', 'darkorange']
lw = 0.2
label_matrix = ['normal', 'tumor']
for color, i, target_name in zip(colors, [0, 1], label_matrix):
    plt.scatter(
        embedding[y == i, 0],
        embedding[y == i, 1],
        color=color,
        alpha=.8,
        lw=lw,
        label=target_name)
fig.tight_layout()
plt.title('UMAP')
fig.savefig('umap_200000.pdf')

Y = fitsne.FItSNE(X_scaled.copy(order='C'))  # max_iter=500)
colors = ['navy', 'darkorange']
lw = 0.2
label_matrix = ['normal', 'tumor']

fig = plt.figure(figsize=(10, 10))
for color, i, target_name in zip(colors, [0, 1], label_matrix):
    plt.scatter(
        Y[y == i, 0],
        Y[y == i, 1],
        color=color,
        alpha=.8,
        lw=lw,
        label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)

fig.tight_layout()
plt.title('FIt-SNE')
fig.savefig('FIT-SNE.pdf')

RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=RANDOM_STATE)

lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print(r2_score_lasso)

alpha = 0.001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(r2_score_enet)

pipeline_optimizer = TPOTClassifier(
    generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
