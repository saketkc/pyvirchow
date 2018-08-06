import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import glob
import os

# Just use 1 GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import load_model
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

from pywsi.io.operations import path_leaf
from pywsi.io.operations import draw_annotation
from pywsi.io.operations import WSIReader


TRAIN_TUMOR_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/tumor'
TRAIN_NORMAL_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/normal'
TRAIN_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/lesion_annotations_json'
TEST_WSI_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/images'
TEST_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/lesion_annotations_json'
MODEL_PATH = '/Z/personal-folders/interns/saket/github/pywsi/scripts/newdropout-sgd-allsamples-keras-improvement-08-0.76.hdf'
PATCH_SIZE = 256


def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """Credits: https://github.com/4QuantOSS/DashIntro/blob/master/notebooks/Tutorial.ipynb
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode('ascii').replace(
        '\n', '')
    return 'data:image/png;base64,{}'.format(encoded)


global model
model = load_model(MODEL_PATH)


def get_samples_from_dir(dir):
    # Just assume all files will have tif extension
    wsis = glob.glob(os.path.join(dir, '*.tif'))
    short_names = []
    for wsi in wsis:
        label = path_leaf(wsi).replace('.tif', '')
        short_names.append({'label': label, 'value': wsi})
    return short_names


ALL_SAMPLES = get_samples_from_dir(TRAIN_TUMOR_DIR) + get_samples_from_dir(
    TRAIN_NORMAL_DIR) + get_samples_from_dir(TEST_WSI_DIR)

app = dash.Dash()
app.layout = html.Div([
    html.Div([ dcc.Dropdown(id='wsi-dropdown', options=ALL_SAMPLES, value='')],
                 style={'width': '20%', 'display': 'block'}),
    html.Div([html.Img(id='thumbnail_plot', src='')], id='plot_div',
             style={'width': '50%', 'display': 'inline-block', 'float': 'left'}),
    html.Div([html.Img(id='thumbnail_plot2', src='')], id='plot_div',
             style={'width': '50%', 'display': 'inline-block', 'float': 'left'}),
])


@app.callback(
    Output(component_id='thumbnail_plot', component_property='src'),
    [Input('wsi-dropdown', 'value')])
def update_output(slide_path):
    slide = WSIReader(slide_path, 40)
    uid = slide.uid

    n_cols = int(slide.width / 256)
    n_rows = int(slide.height / 256)

    thumbnail = slide.get_thumbnail((n_cols, n_rows))
    thumbnail = np.array(thumbnail)
    fig, ax = plt.subplots()
    ax.imshow(thumbnail)
    if 'tumor' in uid:
        # Load annotation
        json_filepath = os.path.join(TRAIN_ANNOTATION_DIR, uid+'.json')
        draw_annotation(json_filepath, 0, 0, 1/256, ax)
    elif 'test' in uid:
        # Load annotation
        json_filepath = os.path.join(TEST_ANNOTATION_DIR, uid+'.json')
        if os.path.isfile(json_filepath):
            draw_annotation(json_filepath, 0, 0, 1/256, ax)
    ax.axis('off')
    out_url = fig_to_uri(fig)
    return out_url

@app.callback(
    Output(component_id='thumbnail_plot2', component_property='src'),
    [Input('wsi-dropdown', 'value')])
def update_output2(slide_path):
    slide = WSIReader(slide_path, 40)
    n_cols = int(slide.width / 256)
    n_rows = int(slide.height / 256)

    thumbnail = slide.get_thumbnail((n_cols, n_rows))
    thumbnail = np.array(thumbnail)
    fig, ax = plt.subplots()
    ax.imshow(thumbnail)
    ax.axis('off')
    out_url = fig_to_uri(fig)
    return out_url


if __name__ == '__main__':
    app.title = 'Virchow: Classification of Histopathology Images'
    app.run_server(host='192.168.221.21', debug=True)
