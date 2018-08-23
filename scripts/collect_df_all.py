"""Collect all dataframes in a folder"""
import os
import glob

import sys
from tqdm import tqdm
import pandas as pd
from pywsi.misc.helpers import order
import joblib

def main(source_df, final_df_path):

    """Specifically collect summary_regions.tsv"""
    master_df = pd.DataFrame()
    source_df = pd.read_table(source_df)
    if not os.path.isfile(final_df_path+'.merged.pickle'):
        all_dfs = source_df['segmented_tsv'].apply(lambda x: pd.read_table(x))
        joblib.dump(all_dfs, final_df_path+'.merged.pickle')
        for df, is_tumor in zip(all_dfs, source_df['is_tumor']):
            df['is_tumor'] = is_tumor

        joblib.dump(all_dfs, final_df_path+'.merged.pickle')
    else:
        all_dfs = joblib.load(final_df_path+'.merged.pickle')
    all_dfs = all_dfs.tolist()
    master_df = pd.concat(all_dfs)
    master_df = master_df.drop(columns=['0'])
    master_df = order(master_df, ['is_tissue', 'is_tumor'])
    master_df.to_csv(final_df_path, sep='\t', index=False,
            header=True)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        sys.stderr.write('In sufficient params. To run: python collect_df.py <folder_path> <final_df_path>\n')
        sys.exit(1)
    source_df, final_df_path = sys.argv[1:]
    main(source_df, final_df_path)
