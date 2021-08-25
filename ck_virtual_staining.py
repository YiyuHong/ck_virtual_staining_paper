import os
import sys
import cv2
import numpy as np
import openslide
import torch
import torchvision.transforms as transforms
from networks import define_G
from skimage.color import rgb2gray
from tifffile import memmap


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


def predict_wsi(input_file_path, output_file_path, model_file_path,
                local_size=1024, margin=256, img_resize_factor=2,
                scale_factor=4):
    if input_file_path is None or not input_file_path:
        sys.stderr.write('Input file path is required.')
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

    he_slide = openslide.open_slide(input_file_path)
    slide_width, slide_height = he_slide.dimensions

    num_w = slide_width // local_size + 1
    num_h = slide_height // local_size + 1

    remains_width = slide_width % local_size
    remains_height = slide_height % local_size

    # ROI_w, ROI_h = ROI_region
    filename = os.path.basename(input_file_path)
    result_name = 'predicted_{}_patch{}_margin{}.tif'.format(filename,
                                                             local_size,
                                                             margin)
    result_path = os.path.join(output_file_path, result_name)
    # result image memory map
    image_file = memmap(result_path,
                        shape=(slide_height // scale_factor,
                               slide_width // scale_factor, 3),
                        dtype='uint8',
                        bigtiff=False)
    image_file[:, :, :] = 255.
    # get interest tile location
    tmp_a = np.ones((num_w, num_h))  #
    iter_list = [[i[0][0],
                  i[0][1]
                  ] for i in np.ndenumerate(tmp_a)]  #
    len_itr = len(iter_list)

    for itr, [iter_w, iter_h] in enumerate(iter_list):

        l, im_l, im_r, r = get_region(iter_w, slide_width, local_size, margin)
        t, im_t, im_b, b = get_region(iter_h, slide_height, local_size, margin)

        he_patch_raw = he_slide.read_region((l, t), 0, (r - l + 1, b - t + 1))
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

        out_t = cv2.resize(out_t,
                           None,
                           fx=(1 / scale_factor * img_resize_factor),
                           fy=(1 / scale_factor * img_resize_factor))

        sl, sr = resize_region(im_l, im_r, scale_factor)
        st, sb = resize_region(im_t, im_b, scale_factor)
        image_file[st:sb + 1,
        sl:sr + 1, :] = out_t

        if itr % 100 == 0:
            print('Done {}/{}'.format(itr, len_itr))
    image_file.flush()
    # print('finish the predict {}'.format())

# Testing settings
# input_file_path = 'data/wsi/HE_S14-03249-3U.svs'
# output_file_path = 'data/result/wsi'
# model_name = 'model_20200708'
# model_file_path = "data/checkpoints/{}.pth".format(model_name)
# predict_wsi(input_file_path, output_file_path, model_file_path)

