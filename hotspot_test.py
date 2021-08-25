import os
import sys

import cv2
import numpy as np
import openslide
import torch
import torchvision.transforms as transforms
from networks import define_G
from skimage.color import rgb2gray
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from tifffile import memmap
import matplotlib.pyplot as plt


def isBG(img, bg_thres, bg_percent):
    gray_img = np.uint8(rgb2gray(img) * 255)
    #    gray_img = img.convert('L')
    white_percent = np.mean((gray_img > bg_thres))

    black_percent = np.mean((gray_img < 255 - bg_thres))

    if black_percent > bg_percent or white_percent > bg_percent \
            or black_percent + white_percent > bg_percent:
        return True
    else:
        return False


def color_deconvolution(_img, channel):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    _img = rgb2hed(_img)
    _img = _img[:, :, channel]
    _img = rescale_intensity(_img, out_range=(0, 1))
    # Img = 1-Img
    _img = np.uint8(_img * 255)
    return _img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_region(grid_x, image_w, grid_w, margin):
    '''
    Return the base and offset pair to read from the image.
    :param grid_x: grid index on the image
    :param image_w: image width (or height)
    :param grid_w: grid width (or height)
    :param margin: margin width (or height)
    :return: the base index and the width on the image to read
    '''
    image_x = grid_x * grid_w

    image_l = min(image_x, image_w - grid_w)
    image_r = image_l + grid_w - 1

    read_l = max(0, image_l - margin)
    read_r = min(image_r + margin, image_w - 1)
    #    read_l = min(image_x - margin_w, image_w - (grid_w + margin_w))
    #    read_r = min(read_l + grid_w + (margin_w << 1), image_w) - 1
    #    image_l = max(0,read_l + margin_w)
    #    image_r = min(image_l + grid_w , image_w) - 1
    return read_l, image_l, image_r, read_r


def resize_region(im_l, im_r, scale_factor):
    sl = im_l // scale_factor
    sw = (im_r - im_l + 1) // scale_factor
    sr = sl + sw - 1
    return sl, sr


def calculate_annotated_region(image_ck_file,
                               image_he_file,
                               coord,
                               bbox,
                               file_label,
                               region,
                               output_file_path,
                               filename,
                               scale_factor,
                               visualize_result=True):
    bbox_rescale = [i // scale_factor for i in bbox]

    ck_img = image_ck_file[bbox_rescale[1]:bbox_rescale[3],
                           bbox_rescale[0]:bbox_rescale[2], :].copy()
    he_img = image_he_file[bbox_rescale[1]:bbox_rescale[3],
                           bbox_rescale[0]:bbox_rescale[2], :].copy()
    mask = cv2.fillPoly(image_he_file,
                        [coord // scale_factor],
                        (0, 0, 0))[bbox_rescale[1]:bbox_rescale[3],
           bbox_rescale[0]:bbox_rescale[2], 0].copy()
    mask = mask == 0

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
    count_stroma = str(stroma_img[mask].tolist()).count('255')
    # count pixel number of tissue
    count_tissue = str(he_binary_img[mask].tolist()).count('255')

    if count_tissue == 0:
        tsp_value = 0  # None
    else:
        tsp_value = count_stroma / count_tissue  # Tsp value

    if visualize_result:
        resclae_poly = [[i[0] - bbox_rescale[0],
                         i[1] - bbox_rescale[1]] for i in coord // scale_factor]
        ck_img = cv2.polylines(ck_img,
                               [np.array(resclae_poly)],
                               True,
                               (255, 0, 0),
                               20).copy()

        he_img = cv2.polylines(he_img,
                               [np.array(resclae_poly)],
                               True,
                               (255, 0, 0),
                               20).copy()

        # tsr img
        env_tissue = np.zeros((*stroma_img.shape, 3), dtype='uint8')
        env_tissue[:, :, 0] = ck_binary_img
        env_tissue[:, :, 1] = stroma_img
        env_tissue[np.sum(env_tissue, axis=2) == 0] = 255

        plt.imsave(os.path.join(output_file_path,
                                filename + '_part{}_CK.jpg'.format(region)),
                   ck_img)
        plt.imsave(os.path.join(output_file_path,
                                filename + '_part{}_HE.jpg'.format(region)),
                   he_img)
        plt.imsave(os.path.join(output_file_path,
                                filename + '_part{}_Tissue.jpg'.format(region)),
                   he_binary_img, cmap = 'gray')
        plt.imsave(os.path.join(output_file_path,
                                filename + '_part{}_Stroma.jpg'.format(region)),
                   stroma_img, cmap = 'gray')
        plt.imsave(os.path.join(output_file_path,
                                filename + '_part{}_Tumor.jpg'.format(region)),
                   ck_binary_img, cmap = 'gray')
        plt.imsave(os.path.join(output_file_path,
                                filename + '_part{}_ENV.jpg'.format(region)),
                   env_tissue)

    with open(os.path.join(output_file_path,
                           filename + '_part{}.txt'.format(region)),
              'w') as f:
        f.write('tsp,label\n')
        f.write(str(tsp_value) + ',' + file_label)
    f.close()


def predict_annotated_wsi(input_file_path, annotation_file_path, output_file_path,
                model_file_path, local_size=1024, margin=256,
                img_resize_factor=2, scale_factor=4, visualize=True):
    if input_file_path is None or not input_file_path:
        sys.stderr.write('Input file path is required.')
        exit(1)
    if annotation_file_path is None or not annotation_file_path:
        sys.stderr.write('Annotation file path is required.')
        exit(1)
    if output_file_path is None or not output_file_path:
        sys.stderr.write('Output file path is required.')
        exit(1)
    if model_file_path is None or not model_file_path:
        sys.stderr.write('Model file path is required.')
        exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_g = define_G('unet', 3, 3)
    net_g.load_state_dict(torch.load(model_file_path))

    filename = os.path.splitext(os.path.basename(input_file_path))[0]
    annotation = np.load(annotation_file_path,
                         allow_pickle=True).item()
    file_label = annotation['label']
    num_region = annotation['num_region']

    for region in range(num_region):
        coord_x = annotation[region]['coord_x']
        coord_y = annotation[region]['coord_y']
        coord = np.array([coord_x, coord_y])
        coord = coord.transpose(1, 0)
        bbox = annotation[region]['bbox']

        he_slide = openslide.open_slide(input_file_path)
        slide_width, slide_height = he_slide.dimensions

        num_w = slide_width // local_size + 1
        num_h = slide_height // local_size + 1

        # ROI_w, ROI_h = ROI_region
        result_ck_name = 'tmp_CK.tif'
        result_ck_path = os.path.join(output_file_path, result_ck_name)
        # result image memory map
        image_ck_file = memmap(result_ck_path,
                            shape=(slide_height // scale_factor,
                                   slide_width // scale_factor, 3),
                            dtype='uint8',
                            bigtiff=False)
        image_ck_file[:, :, :] = 255.

        # HE files
        result_he_name = 'tmp_HE.tif'
        result_he_path = os.path.join(output_file_path, result_he_name)
        # result image memory map
        image_he_file = memmap(result_he_path,
                               shape=(slide_height // scale_factor,
                                      slide_width // scale_factor, 3),
                               dtype='uint8',
                               bigtiff=False)
        image_he_file[:, :, :] = 255.

        # get interest tile location
        tmp_a = np.ones((num_w, num_h))  #
        iter_list = [[i[0][0],
                      i[0][1]
                      ] for i in np.ndenumerate(tmp_a)]  #
        len_itr = len(iter_list)

        for itr, [iter_w, iter_h] in enumerate(iter_list):

            l, im_l, im_r, r = get_region(iter_w, slide_width, local_size,
                                          margin)
            t, im_t, im_b, b = get_region(iter_h, slide_height, local_size,
                                          margin)

            sl, sr = resize_region(im_l, im_r, scale_factor)
            st, sb = resize_region(im_t, im_b, scale_factor)

            # continue the for loop if selected patch
            # is not in annotation area
            # if not ((bbox[0] < im_l and bbox[2] > im_l) or (
            #         bbox[0] < im_r and bbox[2] > im_r)) and \
            #       ((bbox[1] < im_t and bbox[3] > im_t) or (
            #         bbox[1] < im_b and bbox[3] > im_b)):
            #    continue

            if not ((bbox[0] < im_l < bbox[2] or
                    bbox[0] < im_r < bbox[2]) and \
                   (bbox[1] < im_t < bbox[3] or
                    bbox[1] < im_b < bbox[3])):
                continue

            he_patch_raw = he_slide.read_region((l, t), 0,
                                                (r - l + 1, b - t + 1))
            he_patch_raw = np.array(he_patch_raw)[:, :, [0, 1, 2]]

            if isBG(he_patch_raw, 240, 0.95):
                continue

            he_patch_resized = cv2.resize(he_patch_raw,
                                          None,
                                          fx=1 / img_resize_factor,
                                          fy=1 / img_resize_factor)

            he_patch_tensor = transforms.ToTensor()(he_patch_resized)
            he_patch_tensor = he_patch_tensor.view(1, *he_patch_tensor.shape)
            input_ = he_patch_tensor.to(device).type(torch.float32)
            out = net_g(input_)[:, :3, :, :].detach().cpu().numpy().copy()

            out[out > 1] = 1
            out[out < 0] = 0

            out_t = out.squeeze(0).transpose(1, 2, 0)
            out_t = (out_t * 255).astype('uint8')

            out_t = out_t[(im_t - t) // img_resize_factor:(
                           im_b - t + 1) // img_resize_factor,
                          (im_l - l) // img_resize_factor:(
                           im_r - l + 1) // img_resize_factor,
                          :]
            he_t = he_patch_resized[(im_t - t) // img_resize_factor:(
                                     im_b - t + 1) // img_resize_factor,
                                    (im_l - l) // img_resize_factor:(
                                     im_r - l + 1) // img_resize_factor,
                                    :]
            out_t = cv2.resize(out_t,
                               None,
                               fx=(1 / scale_factor * img_resize_factor),
                               fy=(1 / scale_factor * img_resize_factor))
            he_t = cv2.resize(he_t,
                               None,
                               fx=(1 / scale_factor * img_resize_factor),
                               fy=(1 / scale_factor * img_resize_factor))
            image_ck_file[st:sb + 1,
                          sl:sr + 1, :] = out_t
            image_he_file[st:sb + 1,
                          sl:sr + 1, :] = he_t

            if itr % 100 == 0:
                print('Done {}/{}'.format(itr, len_itr))

        calculate_annotated_region(image_ck_file,
                                   image_he_file,
                                   coord,
                                   bbox,
                                   file_label,
                                   region,
                                   output_file_path,
                                   filename,
                                   scale_factor,
                                   visualize_result=visualize)

    image_ck_file.flush()
    image_he_file.flush()

    if os.path.exists(result_ck_path):
        os.remove(result_ck_path)
    if os.path.exists(result_he_path):
        os.remove(result_he_path)

    print('finishing predict the {}'.format(filename))


#Testing settings
# filename = 'S15-20845-3P-low'
# model_name = 'model_20200708'
#
# input_file_path = './data/wsi/hotspot_slide/{}.svs'.format(filename)
# annotation_file_path = './data/wsi/hotspot_annotation/{}.npy'.format(filename)
# output_file_path = './data/result/hotspot'
# model_file_path = "./data/checkpoints/{}.pth".format(model_name)
#
# predict_annotated_wsi(input_file_path,
#                       annotation_file_path,
#                       output_file_path,
#                       model_file_path,
#                       visualize=True)
