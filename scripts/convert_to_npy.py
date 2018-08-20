import os
import sys
import glob
import joblib
import numpy as np
from tqdm import tqdm
from joblib import delayed
from pywsi.misc.parallel import ParallelExecutor
folder = sys.argv[1]
aprun = ParallelExecutor(n_jobs=8)


img_files = glob.glob(os.path.join(folder, '*.mask.joblib.pickle'))
def convert_to_npy(filepath):
    data = joblib.load(filepath)
    filename = filepath.replace('.pickle', '.npy')
    if not os.path.isfile(filename):
        np.save(filename, data)
total = len(img_files)
aprun(total=total)(delayed(convert_to_npy)(f) for f in img_files)

#with tqdm(total=total) as pbar:
#    for _ in Parallel(n_jobs=8)(delayed(convert_to_npy)(f) for f in img_files):
#        pbar.update()
