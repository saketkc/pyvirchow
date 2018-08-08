import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import glob
import os
import joblib

# Just use 1 GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from pywsi.deep_model.plot_utils import plot_blend

TRAIN_TUMOR_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/tumor'
TRAIN_NORMAL_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/normal'
TRAIN_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/training/lesion_annotations_json'
TEST_WSI_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/images'
TEST_ANNOTATION_DIR = '/Z/personal-folders/interns/saket/histopath_data/CAMELYON16/testing/lesion_annotations_json'
PREDICTED_HEATMAP_DIR = '/Z/personal-folders/interns/saket/github/pywsi/data/wsi_heatmap_sgd'
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
    html.Div(
        [dcc.Dropdown(id='wsi-dropdown', options=ALL_SAMPLES, value='')],
        style={
            'width': '20%',
            'display': 'block'
        }),
    html.Div(
        [html.Img(id='thumbnail_plot0', src='', width='100%', height='100%')],
        id='plot_div0',
        style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'left'
        }),
    html.Div(
        [html.Div(id='embed')],
        id='plot_div1',
        style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'left'
        }),
    html.Div(
        [html.Img(id='thumbnail_plot2', src='', width='100%', height='100%')],
        id='plot_div2',
        style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'left'
        }),
    html.Div(
        [html.Img(id='thumbnail_plot3', src='', width='100%', height='100%')],
        id='plot_div3',
        style={
            'width': '50%',
            'display': 'inline-block',
            'float': 'left'
        }),
])


@app.callback(
    Output(component_id='thumbnail_plot0', component_property='src'),
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
        json_filepath = os.path.join(TRAIN_ANNOTATION_DIR, uid + '.json')
        draw_annotation(json_filepath, 0, 0, 1 / 256, ax)
    elif 'test' in uid:
        # Load annotation
        json_filepath = os.path.join(TEST_ANNOTATION_DIR, uid + '.json')
        if os.path.isfile(json_filepath):
            draw_annotation(json_filepath, 0, 0, 1 / 256, ax)
    #ax.axis('off')
    fig.tight_layout()
    out_url = fig_to_uri(fig)
    plt.close('all')
    return out_url


@app.callback(Output('embed', 'children'), [Input('wsi-dropdown', 'value')])
def update_output(slide_path):
    slide = WSIReader(slide_path, 40)
    uid = slide.uid
    if 'tumor' in uid:
        # These servers are being run through deepzoom_multiserver.py script
        # available as part of examples of openslide-python
        port = 5000
    elif 'normal' in uid:
        port = 5001
    elif 'test' in uid:
        port = 5002
    return html.Iframe(
        src='http://192.168.221.21:{}/{}.tif'.format(port, uid),
        width='100%',
        height='480px;')


@app.callback(
    Output(component_id='thumbnail_plot2', component_property='src'),
    [Input('wsi-dropdown', 'value')])
def update_output2(slide_path):
    uid = path_leaf(slide_path).replace('.tif', '')
    pickle_file = os.path.join(PREDICTED_HEATMAP_DIR, uid + '.joblib.pickle')
    slide = WSIReader(slide_path, 40)
    n_cols = int(slide.width / 256)
    n_rows = int(slide.height / 256)

    thumbnail = slide.get_thumbnail((n_cols, n_rows))
    thumbnail = np.array(thumbnail)
    if os.path.isfile(pickle_file):
        thumbnail_predicted = joblib.load(pickle_file)
        fig, ax = plt.subplots()
        plot_blend(thumbnail, thumbnail_predicted, ax, alpha=1)
        #ax.axis('off')
        fig.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.imshow(thumbnail)
        #ax.axis('off')
        fig.tight_layout()
    slide.close()
    out_url = fig_to_uri(fig)
    plt.close('all')
    return out_url


@app.callback(
    Output(component_id='thumbnail_plot3', component_property='src'),
    [Input('wsi-dropdown', 'value')])
def update_output3(slide_path):
    uid = path_leaf(slide_path).replace('.tif', '')
    pickle_file = os.path.join(PREDICTED_HEATMAP_DIR, uid + '.joblib.pickle')
    if os.path.isfile(pickle_file):
        thumbnail_predicted = joblib.load(pickle_file)
        fig, ax = plt.subplots()
        ax.imshow(
            (thumbnail_predicted > 0.75).astype(np.int),
            cmap='gray',
            vmin=0,
            vmax=1)
        #ax.set_title(' (white=tumor, black=not_tumor)')
        #ax.axis('off')
        fig.tight_layout()
    else:
        slide = WSIReader(slide_path, 40)
        n_cols = int(slide.width / 256)
        n_rows = int(slide.height / 256)

        thumbnail = slide.get_thumbnail((n_cols, n_rows))
        thumbnail = np.array(thumbnail)
        fig, ax = plt.subplots()
        ax.imshow(thumbnail)
        ax.axis('off')
        fig.tight_layout()
        slide.close()
    out_url = fig_to_uri(fig)
    plt.close('all')
    return out_url


if __name__ == '__main__':
    app.title = 'Virchow: Classification of Histopathology Images'
    app.run_server(host='192.168.221.21', debug=True)
