import os
import json
from bs4 import BeautifulSoup
import numpy as np


def xmltojson(xml_path, savedir):
    """Specifically designed for the CAMELYON16 xml

    Parameters
    ----------
    xml_path: string
              Path to xml file
    savedir: string
             Path to save json file

    """
    with open(xml_path) as fh:
        soup = BeautifulSoup(fh, 'lxml')
    json_dict = {}
    json_dict['tumor'] = []
    json_dict['normal'] = []
    for annotation in soup.findAll('annotation'):
        tumor_coordinates = []
        normal_coordinates = []
        if annotation['partofgroup'] == '_1':
            for coordinate in annotation.findAll('coordinate'):
                tumor_coordinates.append([
                    int(np.round(float(coordinate['x'])).astype(int)),
                    int(np.round(float(coordinate['y'])).astype(int))
                ])
            json_dict['tumor'].append({
                'name': annotation['name'],
                'vertices': tumor_coordinates
            })

        elif annotation['partofgroup'] == '_0':
            for coordinate in annotation.findAll('coordinate'):
                tumor_coordinates.append([
                    int(np.round(float(coordinate['x'])).astype(int)),
                    int(np.round(float(coordinate['y'])).astype(int))
                ])
            json_dict['tumor'].append({
                'name': annotation['name'],
                'vertices': tumor_coordinates
            })
        elif annotation['partofgroup'] == '_2':
            for coordinate in annotation.findAll('coordinate'):
                normal_coordinates.append([
                    int(np.round(float(coordinate['x'])).astype(int)),
                    int(np.round(float(coordinate['y'])).astype(int))
                ])
            json_dict['normal'].append({
                'name': annotation['name'],
                'vertices': normal_coordinates
            })
        elif annotation['partofgroup'] == 'Exclusion':
            # Exclude
            pass
        elif annotation['partofgroup'] == 'None':
            #continue
            pass
        else:
            raise RuntimeError('Did not find an appropriate group id')
    with open(os.path.join(savedir, xml_path.replace('.xml', '.json')),
              'w') as fw:
        json.dump(json_dict, fw, indent=1)
