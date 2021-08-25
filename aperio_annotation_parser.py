import os
import re
import numpy as np

def ParseQupathAnnotation(path, basepath='./data/wsi/hotspot_annotation'):
    annotation = {}
    files_path = path
    file_name = os.path.splitext(os.path.basename(path))[0]
    annotation[file_name] = {}
    idx = 0

    with open(os.path.join(files_path), 'r') as f:
        file_path = f.readline().replace("\n", '')
        annotation[file_name]['path'] = file_path
        for line1, line2 in zip(f, f):
            line_name = line1.replace("\n", '')
            line_coord = line2.replace("\n", '')
            annotation[file_name][str(idx)] = {}
            annotation[file_name][str(idx)]['class'] = line_name

            coord_x = [int(float(i)) for idx, i in
                       enumerate(re.findall('\d*\.\d*', line_coord)) if
                       idx % 2 == 0]
            coord_y = [int(float(i)) for idx, i in
                       enumerate(re.findall('\d*\.\d*', line_coord)) if
                       idx % 2 == 1]

            annotation[file_name][str(idx)]['coord'] = [min(coord_x),
                                                        min(coord_y),
                                                        max(coord_x),
                                                        max(coord_y)]
            idx += 1

    np.save(os.path.join(basepath,
                         '{}.npy'.format(file_name)),
            annotation)
    print('Anno. {} saved at {}'.format(file_name, basepath))


basepath = './data/wsi/hotspot_annotation'
anno_path = os.path.join(basepath, 'annotation')
ParseQupathAnnotation(anno_path)