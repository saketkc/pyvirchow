import os

TRAIN_TUMOR_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/tumor'
TRAIN_NORMAL_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/normal'
TRAIN_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/lesion_annotations_json'
TEST_WSI_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/images'
PATCH_SAVEDIR = '/Z/personal-folders/interns/saket/github/pyvirchow/data/patch_df/'
TEST_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/lesion_annotations_json'
NOT_AVAIL_IMG = '/Z/personal-folders/interns/saket/github/pyvirchow/pyvirchow/web/assets/No_image_available.png'
PREDICTED_HEATMAP_DIR = {
    'cnn':
    '/Z/personal-folders/interns/saket/github/pyvirchow/data/wsi_heatmap_sgd',
    'rf': '/Z/personal-folders/interns/saket/github/pyvirchow/data/wsi_heatmap_rf'
}
PATCH_SIZE = 256
DATA_DIR = '/Z/personal-folders/interns/saket/github/pyvirchow/data'


TUMOR_VALIDATE_SLIDES = ['tumor_005',
 'tumor_011',
 'tumor_031',
 'tumor_046',
 'tumor_065',
 'tumor_069',
 'tumor_079',
 'tumor_085',
 'tumor_097']

NORMAL_VALIDATE_SLIDES = ['normal_016',
 'normal_020',
 'normal_030',
 'normal_046',
 'normal_057',
 'normal_092',
 'normal_097',
 'normal_098',
 'normal_100',
 'normal_130',
 'normal_136',
 'normal_142',
 'normal_159']
for slide in TUMOR_VALIDATE_SLIDES[::-1]:
    print(slide)
    slide_path = os.path.join(TRAIN_TUMOR_DIR, slide + '.tif')
    df_path = os.path.join(PATCH_SAVEDIR, slide +'.tsv')
    if not os.path.isfile(df_path):
        cmd = 'pyvirchow tif-to-df --tif {} --jsondir {} --savedir {}'.format(slide_path, TRAIN_ANNOTATION_DIR, PATCH_SAVEDIR)
        os.system(cmd)
    df_with_mask_path = df_path.replace('.tsv', '')+'_with_mask.tsv'
    df_with_mask_segmented_path = df_path.replace('.tsv', '')+'_with_mask_segmented.tsv'
    patch_segment_dir = os.path.join(DATA_DIR, 'patch_segmented_{}'.format(slide))
    cmd = 'pyvirchow add-patch-mask-col --df {} --savedf {}'.format(df_path, df_with_mask_path)
    os.system(cmd)

    if not os.path.isfile(df_with_mask_segmented_path.replace('.tsv', '.segmented.tsv')):
        print('segmenting..')
        cmd = 'pyvirchow segment-from-df-fast --df {} --savedir {} --finaldf {}'.format(df_with_mask_path,
                                                                                    patch_segment_dir,
                                                                                    df_with_mask_segmented_path)
        os.system(cmd)
    pickle_file = '{}/{}.joblib.pickle'.format(patch_segment_dir, slide)
    if os.path.isfile(pickle_file):
        continue
    print('heatmap..')
    cmd = 'pyvirchow heatmap-rf --tif {} --savedir {} --df {}'.format(slide_path,
                                                                  patch_segment_dir,
                                                                  df_with_mask_segmented_path)
    os.system(cmd)
