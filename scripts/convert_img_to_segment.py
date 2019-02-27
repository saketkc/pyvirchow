import os
import sys
import glob
import joblib
import numpy as np
from tqdm import tqdm
import joblib
from joblib import delayed
from pywsi.misc.parallel import ParallelExecutor
folder = sys.argv[1]

aprun = ParallelExecutor(n_jobs=8)
def convert_to_npy(filepath):
    data = joblib.load(filepath)
    filename = filepath.replace('.pickle', '.npy')
    if not os.path.isfile(filename):
        np.save(filename, data)
total = len(img_files)
aprun(total=total)(delayed(convert_to_npy)(f) for f in img_files)
