'''
Moving Image Coordinate - Final Moving Image Translation = Registered Coordinates
Coordinate -> (x,y) start from top left
A0 Donghwan Lee
Reference. WSI_Registration_for_local_translation_WDY_191215_ ph.D Wang, ph.D Hong
'''

import math
import os

import cv2
import numpy as np
import openslide
from scipy import signal
from skimage import morphology
from skimage.filters import threshold_local

np.set_printoptions(threshold=np.inf)

from skimage.color import rgb2gray, rgb2hed
from skimage.exposure import rescale_intensity

from multiprocessing import Pool
import tqdm


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def Color_Deconvolution(Img, Channel):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    Img = rgb2hed(Img)
    Img = Img[:, :, Channel]
    Img = rescale_intensity(Img, out_range=(0, 1))
    Img = np.uint8(Img * 255)
    return Img


def Registration_Translation(Target_Img, Moving_Img):
    Target_Img = rgb2gray(Target_Img)
    Moving_Img = rgb2gray(Moving_Img)

    adaptive_thresh = threshold_local(Target_Img, 51, offset=3)
    Target_Img_B = Target_Img < adaptive_thresh

    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    # Target_Img_B = morphology.remove_small_holes(Target_Img_B, 200)
    # Target_Img_B = morphology.binary_dilation(Target_Img_B, square(3))

    adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
    Moving_Img_B = Moving_Img < adaptive_thresh
    Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)  # 200
    # Moving_Img_B = morphology.remove_small_holes(Moving_Img_B, 400)

    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    if np.max(Cross_Corr) != 0:
        Cross_Corr = Cross_Corr / np.max(Cross_Corr)
    else:
        Cross_Corr = Cross_Corr / 1

    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
    # Moving_Img_Translation-> + , plus the tranlation to moving image coord , Moving_Img_Translation-> - minus the tranlation to moving image coord
    Moving_Img_Translation = np.array(ind) - np.array(Moving_Img_B.shape)
    return Target_Img_B, Moving_Img_B, Moving_Img_Translation, Cross_Corr


def Local_Registration_Translation(Target_Img,
                                   Moving_Img):  # Calculate the image Translation
    Target_Img = Color_Deconvolution(Target_Img, 1)
    Moving_Img = Color_Deconvolution(Moving_Img, 2)
    #    Target_Img = rgb2gray(Target_Img)
    #    Moving_Img = rgb2gray(Moving_Img)
    #    adaptive_thresh = threshold_local(Target_Img, 51, offset=3)
    adaptive_thresh = threshold_local(Target_Img, 151, offset=10)
    Target_Img_B = Target_Img < adaptive_thresh
    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    Target_Img_B = Target_Img_B.astype(int)
    h, w = Target_Img_B.shape
    Target_Img_B[Target_Img_B == 0] = -1

    adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
    Moving_Img_B = Moving_Img < adaptive_thresh
    Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)  # 200
    Moving_Img_B = Moving_Img_B.astype(int)
    h, w = Moving_Img_B.shape
    Moving_Img_B[Moving_Img_B == 0] = -1

    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    if np.max(Cross_Corr) != 0:
        Cross_Corr = Cross_Corr / np.max(Cross_Corr)
    else:
        Cross_Corr = Cross_Corr / 1

    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
    # Moving_Img_Translation-> + , plus the tranlation to moving image coord , Moving_Img_Translation-> - minus the tranlation to moving image coord
    Moving_Img_Translation = np.array(ind) - (np.array(Moving_Img_B.shape) - 1)
    return Moving_Img_Translation


def PercentBackground(Img, BG_Thres):
    if len(Img.shape) > 2:
        Img = rgb2gray(Img)
        Img = np.uint8(Img)
    White_Percent = np.mean((Img > BG_Thres))
    return White_Percent


def color_deconvolution(Img, Channel):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    Img = rgb2hed(Img)
    Img = Img[:, :, Channel]
    Img = rescale_intensity(Img, out_range=(0, 1))
    Img = np.uint8(Img * 255)
    return Img


def cal_stroma_ratio(he_img, ck_img):
    he_gray_img = rgb2gray(he_img)
    # color deconvolution with DAB channel
    ck_src_dab_img = color_deconvolution(ck_img, 2)
    # binary ck image # 50
    ck_binary_img = (ck_src_dab_img > 50) * 255
    # binarize he image for extract tissue area
    he_binary_img = (he_gray_img < 200) * 255
    # get stroma image
    stroma_img = he_binary_img.copy()
    stroma_img[ck_binary_img == 255] = 0
    # count pixel number of stroma
    count_stroma = str(stroma_img.tolist()).count('255')
    # count pixel number of tissue
    count_tissue = str(he_binary_img.tolist()).count('255')
    if count_tissue == 0:
        tsp_value = 0  # None
    else:
        tsp_value = count_stroma / count_tissue  # Tsp value
    return tsp_value


def registraion_map(arg):
    global CK_org_image
    global HE_org_image

    i = arg[0]
    j = arg[1]
    Global_Translation_h = arg[2]
    Global_Translation_w = arg[3]
    Local_size_w = arg[4]
    Local_size_h = arg[5]
    bound = arg[6]

    HE_start_h = bound + i * Local_size_h
    HE_start_w = bound + j * Local_size_w

    CK_start_h = HE_start_h - Global_Translation_h
    CK_start_w = HE_start_w - Global_Translation_w

    HE_local_image = np.array(
        HE_org_image.read_region((HE_start_w - 512, HE_start_h - 512), 0,
                                 (2048, 2048)))[:, :, :3]
    CK_local_image = np.array(
        CK_org_image.read_region((CK_start_w - 512, CK_start_h - 512), 0,
                                 (2048, 2048)))[:, :,
                     :3]  # cut the local image 10240*10240
    #    HE_local_image = np.array(HE_org_image.read_region((HE_start_w , HE_start_h), 0, (10240, 10240)))[:, :,:3]
    #    CK_local_image = np.array(CK_org_image.read_region((CK_start_w , CK_start_h), 0, (10240, 10240)))[:, :,:3] # cut the local image 10240*10240

    local_translation = Local_Registration_Translation(HE_local_image,
                                                       CK_local_image)
    p_bg = PercentBackground(CK_local_image, 220)
    return i, j, p_bg, local_translation[0], local_translation[
        1], Global_Translation_h, Global_Translation_w


def local_transition_map_cal(reg_map_result, axis=0, grid_size=10):
    '''
    local_transition_map_cal
    1. cal meadian values of transition values by 10*10 kernel
    2. interpolate nan by mean(3*3 kernel)

    axis = 0 transition h value
    axis = w transition h value
    '''
    transition_ = reg_map_result[:, :, axis]
    tmp_ = transition_.copy()

    h, w = tmp_.shape
    tmp_ = tmp_.reshape(h // grid_size, grid_size, w // grid_size, grid_size)
    tmp_ = tmp_.transpose((0, 2, 1, 3))
    tmp_ = tmp_.reshape(h // grid_size, w // grid_size, -1)

    h, w, len_val = tmp_.shape
    null_mask = np.isnan(tmp_)
    null_mask01 = null_mask.sum(axis=-1) > len_val - 30  ##

    tmp_0 = np.nanmedian(tmp_, -1)
    tmp_0[null_mask01] = np.nan

    n_h, n_w = np.where(np.isnan(tmp_0))

    for i, j in zip(n_h, n_w):
        '''
        interpolation with 3by3 window kernel
        '''

        cent_h = i  # n_h[0]
        cent_w = j  # n_w[0]

        left_cent_w = cent_w if cent_w == 0 else cent_w - 1
        left_cent_h = cent_h if cent_h == 0 else cent_h - 1

        right_cent_w = cent_w + 2
        right_cent_h = cent_h + 2

        #    print(dd[left_cent_h:right_cent_h,left_cent_w:right_cent_w])
        int_window = tmp_0[left_cent_h:right_cent_h, left_cent_w:right_cent_w]
        int_window_val = np.nanmean(int_window)
        tmp_0[cent_h, cent_w] = int_window_val

    transition_value = np.kron(tmp_0,
                               np.ones((grid_size, grid_size), dtype=int))
    return transition_value


def GetRegistrationCoordinate(patient_id=None,
                              wsi_path='./data/wsi/train_slide',
                              Downsample_Times=32, n_workers=7):
    if patient_id is None or not patient_id:
        sys.stderr.write('Slide is required.')
        exit(1)
    registration_path = os.path.join(wsi_path, patient_id)
    if os.path.exists('{}/{}-Local-Regeistration.npy'.format(registration_path,
                                                             patient_id)):
        return ('Local-Regeistration already done')

    global CK_org_image
    global HE_org_image

    print('RUN {}'.format(patient_id))
    Target_Slide_Path = os.path.join(wsi_path, 'HE_' + patient_id + '.svs')
    Moving_Slide_Path = os.path.join(wsi_path, 'CK_' + patient_id + '.svs')

    Target_P = openslide.open_slide(Target_Slide_Path)
    Moving_P = openslide.open_slide(Moving_Slide_Path)

    Level = int(math.log2(Downsample_Times)) - 1

    if Level >= len(Target_P.level_dimensions):
        Level = len(Target_P.level_dimensions) - 1

    Downsample_Times = math.ceil(
        Target_P.level_dimensions[0][0] // Target_P.level_dimensions[Level][0])
    Target_Downsampled = np.array(
        Target_P.read_region((0, 0), Level, Target_P.level_dimensions[Level]))[
                         :, :, :3]
    Moving_Downsampled = np.array(
        Moving_P.read_region((0, 0), Level, Moving_P.level_dimensions[Level]))[
                         :, :, :3]

    Target_Downsampled_B, Moving_Downsampled_B, G_Moving_Img_Translation, _ = Registration_Translation(
        Target_Downsampled,
        Moving_Downsampled)  # Calculate the global translation
    G_Moving_Img_Translation = G_Moving_Img_Translation * Downsample_Times
    print(f"Global Moving Image Translation: {G_Moving_Img_Translation}")
    HE_org_image = Target_P
    CK_org_image = Moving_P
    Global_Translation = G_Moving_Img_Translation
    Slide_ID = patient_id
    Local_size_h = 1024
    Local_size_w = 1024
    # global wsi_path

    if not os.path.exists(registration_path):  # create the cut image folder
        os.makedirs(registration_path)

    [org_w, org_h] = CK_org_image.level_dimensions[0]
    bound = 2000  # delete the bound. hulistic number
    num_w = (org_w - bound * 2) // Local_size_w  # number of width
    num_h = (org_h - bound * 2) // Local_size_h  # number of height

    A = np.ones((num_h, num_w))
    total_len = num_w * num_h

    iter_list = [[i[0][0],
                  i[0][1],
                  Global_Translation[0],
                  Global_Translation[1],
                  Local_size_w,
                  Local_size_h,
                  bound
                  ] for i in np.ndenumerate(A)]

    with Pool(n_workers) as p:
        r1 = list(
            tqdm.tqdm(p.imap(registraion_map, iter_list), total=total_len))
    np.save(
        '{}/{}-Local-Regeistration.npy'.format(registration_path, patient_id),
        r1)


def MakePatchImage(patient_id=None, wsi_path='./data/wsi/train_slide',
                   Downsample_Times=32, patch_name='patch_512'):
    print(patient_id)
    registration_path = os.path.join(wsi_path, patient_id)
    if not os.path.exists(
            os.path.join(registration_path, 'CK_{}/'.format(patch_name))):
        os.mkdir(os.path.join(registration_path, 'CK_{}/'.format(patch_name)))
    if not os.path.exists(
            os.path.join(registration_path, 'HE_{}/'.format(patch_name))):
        os.mkdir(os.path.join(registration_path, 'HE_{}/'.format(patch_name)))

    Target_Slide_Path = os.path.join(wsi_path, 'HE_' + patient_id + '.svs')
    Moving_Slide_Path = os.path.join(wsi_path, 'CK_' + patient_id + '.svs')

    Target_P = openslide.open_slide(Target_Slide_Path)
    Moving_P = openslide.open_slide(Moving_Slide_Path)

    r1 = np.load(os.path.join(registration_path,
                              '{}-Local-Regeistration.npy'.format(patient_id)))
    rr = np.array(r1)
    num_h, num_w, _, _, _, _, _ = rr.max(0)

    num_h = int(num_h + 1)
    num_w = int(num_w + 1)

    total_len = int(num_h * num_w)
    reg_map_result = np.ones((int(num_h), int(num_w), 3))

    for idx in range(total_len):
        i = int(rr[idx, 0])
        j = int(rr[idx, 1])
        percent_bg = rr[idx, 2]
        local_translation_h = rr[idx, 3]
        local_translation_w = rr[idx, 4]

        global_translation_h = rr[idx, 5]
        global_translation_w = rr[idx, 6]

        reg_map_result[i, j, 0] = local_translation_w + global_translation_w
        reg_map_result[i, j, 1] = local_translation_h + global_translation_h

        reg_map_result[i, j, 2] = percent_bg

    grid_size = 10
    num_h_pad = grid_size - num_h % grid_size
    num_w_pad = grid_size - num_w % grid_size

    if num_h_pad == grid_size:
        num_h_pad = 0
    if num_w_pad == grid_size:
        num_w_pad = 0

    reg_map_result = np.pad(reg_map_result,
                            ((0, num_h_pad),
                             (0, num_w_pad), (0, 0)),
                            'constant',
                            constant_values=(1))
    bg_mask = reg_map_result[:, :, 2] < 0.00  # get background map
    # aggregate height & low
    transition_value_w = local_transition_map_cal(reg_map_result, 0)
    transition_value_h = local_transition_map_cal(reg_map_result, 1)

    Local_size = 512
    local_registration_size = 1024

    [org_w, org_h] = Target_P.level_dimensions[
        0]  ############################ warning
    bound = 1000  # delete the bound. hulistic number

    row_bg_idx, col_bg_idx = np.where(bg_mask)
    row_not_bg_idx, col_not_bg_idx = np.where(~bg_mask)  #

    patch_info = {}
    for not_gb_index in zip(row_not_bg_idx, col_not_bg_idx):
        #            break
        i = not_gb_index[0]
        j = not_gb_index[1]

        bg_ratio = reg_map_result[:, :, 2][i, j]
        transition_value_h_i = transition_value_h[i, j]
        transition_value_w_i = transition_value_w[i, j]

        CK_start_h = int(local_registration_size * i - transition_value_h_i)
        CK_start_w = int(local_registration_size * j - transition_value_w_i)

        HE_start_h = local_registration_size * i
        HE_start_w = local_registration_size * j

        HE_local_image = np.array(
            Target_P.read_region((HE_start_w, HE_start_h), 0, (
                local_registration_size, local_registration_size)))[:, :,
                         :3]  # cut the local image 10240*10240
        CK_local_image = np.array(
            Moving_P.read_region((CK_start_w, CK_start_h), 0, (
                local_registration_size, local_registration_size)))[:, :, :3]

        for x in [0, 512]:
            for y in [0, 512]:
                subname = '{}_{}_{}'.format(patient_id, CK_start_h + y,
                                            CK_start_w + x)
                ck_patch_path = os.path.join(wsi_path, patient_id,
                                             'CK_{}/CK_{}.jpg'.format(
                                                 patch_name, subname))
                he_patch_path = os.path.join(wsi_path, patient_id,
                                             'HE_{}/HE_{}.jpg'.format(
                                                 patch_name, subname))

                cut_CK = CK_local_image[y:y + Local_size, x:x + Local_size, :]
                cut_HE = HE_local_image[y:y + Local_size, x:x + Local_size, :]
                cv2.imwrite(ck_patch_path, cut_CK[:, :, [2, 1, 0]])
                cv2.imwrite(he_patch_path, cut_HE[:, :, [2, 1, 0]])

                bg_white_ratio = PercentBackground(cut_HE, 220)
                ratio = cal_stroma_ratio(cut_HE, cut_CK)
                he_size = os.path.getsize(he_patch_path)
                ck_size = os.path.getsize(ck_patch_path)

                patch_info[subname] = {}
                patch_info[subname]['ck_path'] = ck_patch_path
                patch_info[subname]['he_path'] = he_patch_path
                patch_info[subname]['size'] = {'ck':ck_size, 'he':he_size}
                patch_info[subname]['tsr'] = ratio
                patch_info[subname]['bgratio'] = bg_white_ratio

    np.save(os.path.join(wsi_path,
                         patient_id, '{}_patch_info'.format(patient_id)),
            patch_info)
    print('Finished to make patch image ')


# GetRegistrationCoordinate(patient_id='S14-05717-3O',
#                           wsi_path='./data/wsi/train_slide',
#                           Downsample_Times=32, n_workers=7)
#
# MakePatchImage(patient_id='S14-05717-3O',
#                wsi_path='./data/wsi/train_slide',
#                Downsample_Times=32,
#                patch_name='patch_images_512')
