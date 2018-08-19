"""Collect all dataframes in a folder"""
import os
import glob

import sys
from tqdm import tqdm
import pandas as pd

def main(folder_path, final_df_path):
    """Specifically collect summary_regions.tsv"""
    path = os.path.join(folder_path, '*.segmented_summary.tsv')
    for index, f in enumerate(tqdm(glob.glob(path))):
        if index == 0:
            df = pd.read_table(f)
        else:
            df = pd.concat([df, pd.read_table(f)])
    df.to_csv(final_df_path, sep='\t', index=False,
              header=True)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        sys.stderr.write('In sufficient params. To run: python collect_df.py <folder_path> <final_df_path>\n')
        sys.exit(1)
    folder_path, final_df_path = sys.argv[1:]
    main(folder_path, final_df_path)
