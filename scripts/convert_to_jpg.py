import os
import sys
import glob
import joblib
from scipy.misc import imsave
from tqdm import tqdm
folder = sys.argv[1]

img_files = glob.glob(os.path.join(folder, '*.img.joblib.pickle'))
for f in tqdm(img_files):
    data = joblib.load(f)
    filename = f.replace('.pickle', '.jpg')
    if not os.path.isfile(filename):
        imsave(filename, data)
