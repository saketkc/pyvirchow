import os
import pandas as pd
from tqdm import tqdm

SOURCE_DIR = '/mnt/disks/data/histopath_data/pyvirchow-out-df-patches/'
DEST_DIR = '/mnt/disks/data/images-for-retrain/validation/'


df = pd.read_table('/home/saketkc/github/pyvirchow/data/df/validate_df_with_mask.tsv')
with tqdm(total=len(df.index)) as pbar:
    for idx, row in df.iterrows():
        filepath = row['img_path'].replace('.pickle', '.jpg').replace('validate6', 'validate')
        if row['is_tumor']:
            os.symlink(filepath, os.path.join(DEST_DIR, 'tumor', os.path.basename(filepath)))
        else:
            os.symlink(filepath, os.path.join(DEST_DIR, 'normal', os.path.basename(filepath)))
        pbar.update()
