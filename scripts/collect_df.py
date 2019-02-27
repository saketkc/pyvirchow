"""Collect all dataframes in a folder"""
import os
import glob

import sys
from tqdm import tqdm
import pandas as pd
from pywsi.misc.helpers import order

def main(source_df, final_df_path):
    """Specifically collect summary_regions.tsv"""
    master_df = pd.DataFrame()
    source_df = pd.read_table(source_df)
    total = len(source_df.index)
    with tqdm(total=total) as pbar:
        for idx, row in source_df.iterrows():
            df = pd.read_table(row['segmented_tsv'])
            df['is_tumor'] = row['is_tumor']
            master_df = pd.concat([master_df, df])
            pbar.update()
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
