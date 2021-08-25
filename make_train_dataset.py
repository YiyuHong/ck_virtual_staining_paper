import glob
import os
import random
import shutil

import numpy as np


def load_patch_info(raw_patch_info):
    patch_info = {}
    for i in raw_patch_info:
        slidename = os.path.dirname(i).split('/')[-1]
        if slidename in ['S08-22358', 'S08-19823', 'S14-05673-3S',
                         'S14-03249-3U']:
            continue
        _info = np.load(i, allow_pickle=True).item()
        patch_info.update(_info)
    return patch_info


def save_logic(bgratio=None,
               tsr=None,
               prob=None,
               threshold_bgratio=None,
               threshold_tsr=None,
               threshold_prob=None):
    global cnt
    global intr_cnt

    if bgratio < threshold_bgratio and \
            tsr > threshold_tsr and \
            prob <= threshold_prob[0]:
        cnt += 1
        intr_cnt += 1
        return True
    elif bgratio < threshold_bgratio and \
            tsr < threshold_tsr and \
            prob <= threshold_prob[1]:
        cnt += 1
        return True
    elif bgratio > threshold_bgratio and \
            tsr >= 0 and \
            prob < threshold_prob[2]:
        cnt += 1
        return True
    else:
        return False


def copypatch(_info,
              save_dir,
              threshold_bgratio,
              threshold_tsr,
              threshold_prob):
    global cnt
    global intr_cnt

    ckdir = _info['ck_path']
    hedir = _info['he_path']
    file_size = min(_info['size']['ck'], _info['size']['he'])
    tsr = _info['tsr']
    bgratio = _info['bgratio']
    prob = random.random()

    if save_logic(bgratio, tsr, prob,
                  threshold_bgratio,
                  threshold_tsr,
                  threshold_prob):
        shutil.copy(ckdir,
                    '{}/CK_images/{}'.format(save_dir, os.path.basename(ckdir)))
        shutil.copy(hedir,
                    '{}/HE_images/{}'.format(save_dir, os.path.basename(hedir)))
    else:
        pass

    if cnt % 1000 == 0:
        print('saved image : {}'.format(cnt))
        print('intr image : {}'.format(intr_cnt))
        if cnt == 0:
            print('intr image ratio : 0')
        else:
            print('intr image ratio : {}'.format(intr_cnt / cnt))





def makedataset(save_dir,
                threshold_bgratio=0.9,
                threshold_tsr=0.05,
                threshold_prob=[1, 0.1, 0.01]):
    global cnt
    global intr_cnt
    cnt = 0
    intr_cnt = 0

    raw_patch_info = sorted(glob.glob('./data/wsi/train_slide/*/*_patch_info*'))
    patch_info = load_patch_info(raw_patch_info)

    pair_patch_list = list(patch_info.keys())
    np.random.shuffle(pair_patch_list)

    if not os.path.exists(save_dir):
        os.mkdir(os.path.join(save_dir, 'CK_images'))
        os.mkdir(os.path.join(save_dir, 'CK_images'))

    for file_name in pair_patch_list:
        copypatch(patch_info[file_name],
                  save_dir,
                  threshold_bgratio,
                  threshold_tsr,
                  threshold_prob)


# makedataset(save_dir='./data/train',
#             threshold_bgratio=0.9,
#             threshold_tsr=0.05,
#             threshold_prob=[1, 0.1, 0.01])