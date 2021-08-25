
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed
from skimage.color import rgb2gray
    
def Color_Deconvolution(Img):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    Img = rgb2hed(Img)
    Img = Img[:, :, :]
    #Img = rescale_intensity(Img, out_range=(0, 1))
    # Img = 1-Img
    #Img = np.uint8(Img * 255)
    return Img


class StainingDataset(Dataset):
    def __init__(self, dataset_dir=None, transform=True, crop=True):
        self.CK_list = sorted(glob.glob('{}/CK_images/*'.format(dataset_dir)))
        self.HE_list = sorted(glob.glob('{}/HE_images/*'.format(dataset_dir)))
        self.transform = transform
        self.crop = crop

        print( 'Number of {} CK images : {}'.format(dataset_dir,
                                                    len(self.CK_list)))
        print( 'Number of {} HE images : {}'.format(dataset_dir,
                                                    len(self.HE_list)))

    def __len__(self):
        if len(self.CK_list) != len(self.HE_list):
            print('differnt length between Ck-HE images')
            assert True

        return len(self.CK_list)

    def __getitem__(self,idx):

#        print(idx)
        HE_path = self.HE_list[idx]
        CK_path = self.CK_list[idx]
        file_name = HE_path.split('/')[-1].split('.')[0]

        if CK_path.split('/')[-1].replace('CK_','') != HE_path.split('/')[-1].replace('HE_', ''):
            print('mismatch file names CK-HE')
            assert True

        HE_image = cv2.imread(HE_path)[:,:,[2,1,0]]
        HE_image = cv2.resize(HE_image,
                              None,
                              fx=0.5,
                              fy=0.5)
        HE_image = transforms.ToPILImage()(HE_image)

        CK_image = cv2.imread(CK_path)[:,:,[2,1,0]]
        CK_image = cv2.resize(CK_image,
                              None,
                              fx=0.5,
                              fy=0.5)
        CK_image = transforms.ToPILImage()(CK_image)

        if self.transform == True and \
        random.random() <= 1.0:
            # random crop
            if self.crop == True:
                crop = transforms.RandomResizedCrop(256)
                i, j, h, w = crop.get_params(CK_image,
                     #scale=(0.08, 1.0), #0.9 0.1
                     #ratio=(0.75, 1.333)) #0.9 1.1
                     scale=(0.9, 1.0),
                     ratio=(0.9, 1.1))
                HE_image = TF.crop(HE_image, i, j, h, w)
                CK_image = TF.crop(CK_image, i, j, h, w)

            else:
                pass

            # flip
            if random.random() > 0.5:
                HE_image = TF.hflip(HE_image)
                CK_image = TF.hflip(CK_image)

            if random.random() > 0.5:
                HE_image = TF.vflip(HE_image)
                CK_image = TF.vflip(CK_image)

            # jitter
            HE_image = transforms.ColorJitter(brightness=0.25,#0.25
                                              contrast=0.75,#0.75
                                              saturation=0.25,#0.25
                                              hue=0.1)(HE_image)

        HE_image = transforms.ToTensor()(HE_image)
        CK_image = transforms.ToTensor()(CK_image)

        staining_data = {'filname' : file_name, 'CK_image' : CK_image,'HE_image' : HE_image}
        return staining_data


class StainingDatasetAux(Dataset):
    def __init__(self, dataset_dir=None, transform=None, crop=True):
        self.CK_list = sorted(glob.glob('{}/CK_images/*'.format(dataset_dir)))
        self.HE_list = sorted(glob.glob('{}/HE_images/*'.format(dataset_dir)))
        self.transform = transform
        self.crop = crop

        print( 'Number of {} CK images : {}'.format(dataset_dir,len(self.CK_list)))
        print( 'Number of {} HE images : {}'.format(dataset_dir,len(self.HE_list)))

    def __len__(self):
        if len(self.CK_list) != len(self.HE_list):
            print('differnt length between Ck-HE images')
            assert True

        return len(self.CK_list)

    def __getitem__(self,idx):
        HE_path = self.HE_list[idx]
        CK_path = self.CK_list[idx]
        file_name = HE_path.split('/')[-1].split('.')[0]

        if CK_path.split('/')[-1].replace('CK_','') != HE_path.split('/')[-1].replace('HE_', ''):
            print('mismatch file names CK-HE')
            assert True

        HE_image = cv2.imread(HE_path)[:,:,[2,1,0]]
        HE_image = cv2.resize(HE_image,
                              None,
                              fx=0.5,
                              fy=0.5)
        HE_image = transforms.ToPILImage()(HE_image)

        CK_image = cv2.imread(CK_path)[:,:,[2,1,0]]
        CK_bin_image = Color_Deconvolution(CK_image.copy())

        CK_image = cv2.resize(CK_image,
                              None,
                              fx=0.5,
                              fy=0.5)
        CK_bin_image = cv2.resize(CK_bin_image,
                                  None,
                                  fx=0.5,
                                  fy=0.5)
        CK_image = transforms.ToPILImage()(CK_image)
        
        if self.transform == True and \
        random.random() <= 1.0:
            #  random crop
            if self.crop == True:
                crop = transforms.RandomResizedCrop(256)
                i, j, h, w = crop.get_params(CK_image,
#                     scale=(0.08, 1.0), #0.9
#                     ratio=(0.75, 1.1333)) #0.9
                     scale=(0.9, 1.0), #0.
                     ratio=(0.9, 1.1)) #0.9
                HE_image = TF.crop(HE_image, i, j, h, w)
                CK_image = TF.crop(CK_image, i, j, h, w)
                CK_bin_image = CK_bin_image[i:i+h,j:j+w,:]
            else:
                pass

            # flip
            if random.random() > 0.5:
                HE_image = TF.hflip(HE_image)
                CK_image = TF.hflip(CK_image)
                #CK_bin_image = TF.hflip(CK_bin_image)
                CK_bin_image = np.flip(CK_bin_image, 1)
            if random.random() > 0.5:
                HE_image = TF.vflip(HE_image)
                CK_image = TF.vflip(CK_image)
                #CK_bin_image = TF.vflip(CK_bin_image)
                CK_bin_image = np.flip(CK_bin_image, 0)
            # jitter
            HE_image = transforms.ColorJitter(brightness=0.25,# 0.25
                                              contrast=0.75, # 0.75
                                              saturation=0.25, # 0.25
                                              hue=0.1)(HE_image) # 0.1

        HE_image = transforms.ToTensor()(HE_image)
        CK_image = transforms.ToTensor()(CK_image)
        CK_bin_image = torch.tensor(np.array(CK_bin_image))

        staining_data = {'filname' : file_name,'HE_image' : HE_image, 'CK_image' : CK_image, 'CK_bin_image' : CK_bin_image}
        return staining_data

