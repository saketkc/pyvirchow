import os

TRAIN_TUMOR_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/tumor'
TRAIN_NORMAL_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/normal'
TRAIN_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/lesion_annotations_json'
TEST_WSI_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/images'
PATCH_SAVEDIR = '/Z/personal-folders/interns/saket/github/pywsi/data/patch_df/'
TEST_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/lesion_annotations_json'
NOT_AVAIL_IMG = '/Z/personal-folders/interns/saket/github/pywsi/pywsi/web/assets/No_image_available.png'
PREDICTED_HEATMAP_DIR = {
    'cnn':
    '/Z/personal-folders/interns/saket/github/pywsi/data/wsi_heatmap_sgd',
    'rf': '/Z/personal-folders/interns/saket/github/pywsi/data/wsi_heatmap_rf'
}
PATCH_SIZE = 256
DATA_DIR = '/Z/personal-folders/interns/saket/github/pywsi/data'


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
NORMAL_VALIDATE_SLIDES = ['normal_159']
for slide in TUMOR_VALIDATE_SLIDES[::-1]:
    slide_path = os.path.join(TRAIN_TUMOR_DIR, slide + '.tif')
    df_path = os.path.join(PATCH_SAVEDIR, slide +'.tsv')
    df_with_mask_path = df_path.replace('.tsv', '')+'_with_mask.tsv'
    df_with_mask_segmented_path = df_path.replace('.tsv', '')+'_with_mask_segmented.tsv'
    pickle_file = '/Z/personal-folders/interns/saket/github/pywsi/data/wsi_heatmap_sgd/{}.joblib.pickle'.format(slide)
    if os.path.isfile(pickle_file):
        continue
    cmd = 'pywsi heatmap --indir {} --jsondir /Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/lesion_annotations_json --savedir /Z/personal-folders/interns/saket/github/pywsi/data/wsi_heatmap_sgd'.format(slide_path)
    os.system(cmd)
