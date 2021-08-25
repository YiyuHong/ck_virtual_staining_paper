import argparse
import logging.handlers
import os
import sys
import datetime

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.append('./src')
from networks import define_D, define_G, GANLoss, get_scheduler, update_learning_rate
from helper import StainingDataset, StainingDatasetAux


def TorchRgb2hed(rgb, trans_mat):
    rgb = rgb.squeeze().permute(1, 2, 0)
    rgb = rgb + 2
    stain = -torch.log(rgb.view(-1, 3))
    stain = torch.matmul(stain, trans_mat, out=None)
    return stain.view(rgb.shape)


# Training settings
def train(run_info='model', dataset_path='data/train', img_per_epoch =10000,
          batch_size=1, model='unet', input_nc=3, output_nc=3, ngf=64, ndf=64,
          epoch_count=1, niter=100, niter_decay=100, lr=0.0002,
          lr_policy='lambda', lr_decay_iters=50, beta1=0.5, cuda=True,
              threads=4,seed=123,lamb=10,lamb_hed=0.9,hed_normalize=False):
    print(run_info)
    if not os.path.exists("./data/checkpoints"):
        os.mkdir("./data/checkpoints")

    if not os.path.exists(os.path.join("./data/checkpoints", run_info)):
        os.mkdir(os.path.join("./data/checkpoints", run_info))

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if cuda else "cpu")

    log = logging.getLogger('staining_log')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler('./data/checkpoints/{}/log.txt'.format(run_info))
    streamHandler = logging.StreamHandler()
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    #
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    log.info('info: {} \n \n'.format('Training H&E -> CK'))
    log.info('info: Loading datasets')
    n_workers = threads
    batch_size = batch_size
    img_per_epoch = img_per_epoch

    staining_train_dataset = StainingDatasetAux(
        dataset_dir=dataset_path,
        transform=True,
        crop=True
    )
    dataset_train_loader = DataLoader(staining_train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=n_workers,
                                      )

    log.info('info: Building models')

    net_g = define_G(model,
                     input_nc,
                     output_nc,
                     gpu_id=device)

    net_d = define_D(input_nc + output_nc,
                     ndf,
                     netD='basic',
                     gpu_id=device)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(),
                             lr=lr,
                             betas=(beta1, 0.999))

    optimizer_d = optim.Adam(net_d.parameters(),
                             lr=lr,
                             betas=(beta1, 0.999))

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    net_g_scheduler = get_scheduler(optimizer_g, lr_policy, epoch_count,
                                    niter, niter_decay, lr_decay_iters)
    net_d_scheduler = get_scheduler(optimizer_d, lr_policy, epoch_count,
                                    niter, niter_decay, lr_decay_iters)

    rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                             [0.07, 0.99, 0.11],
                             [0.27, 0.57, 0.78]])
    hed_from_rgb = linalg.inv(rgb_from_hed)
    hed_from_rgb = torch.Tensor(hed_from_rgb).cuda()


    for epoch in range(1, 201):

        loss_d_list = []
        loss_g_list = []
        loss_g_gan_list = []
        loss_g_l1_list = []
        loss_g_hed_l1_list = []

        for iteration, batch in enumerate(dataset_train_loader):
            real_a = batch['HE_image'].to(device).type(torch.float32)
            real_b = batch['CK_image'].to(device).type(torch.float32)
            real_b_hed = batch['CK_bin_image'].to(device).type(torch.float32) / 255.

            # generate fake image
            fake_b = net_g(real_a)
            #        real_a.shape
            ######################
            # (1) Update D network
            ######################
            optimizer_d.zero_grad()

            # predict with fake on Discriminator and calculate loss
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)  # if true = False

            # predict with real on Discriminator and calculate loss
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)  # if true = True

            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5  # average Discriminator losse

            loss_d.backward()
            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b)
            real_b_hed = real_b_hed.squeeze()
            fake_hed = TorchRgb2hed(fake_b, hed_from_rgb)

            if hed_normalize:
                fake_hed -= fake_hed.min(1, keepdim=True)[0]
                fake_hed /= fake_hed.max(1, keepdim=True)[0]
                real_b_hed -= real_b_hed.min(1, keepdim=True)[0]
                real_b_hed /= real_b_hed.max(1, keepdim=True)[0]

            loss_hed_l1 = criterionL1(fake_hed[:, :, :], real_b_hed[:, :, :])
            loss_g = loss_g_gan + loss_g_l1 * lamb + loss_hed_l1 * lamb_hed
            loss_g.backward()
            optimizer_g.step()

            loss_d_list.append(loss_d.item())
            loss_g_list.append(loss_g.item())
            loss_g_gan_list.append(loss_g_gan.item())
            loss_g_l1_list.append(loss_g_l1.item())
            loss_g_hed_l1_list.append(loss_hed_l1)

            if iteration % 1000 == 0:
                log.info(
                    'Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}- GAN: {:.4f}, L1Loss: {:.4f}, hed_Loss: {:.4f} '.format(
                        epoch,
                        iteration,
                        len(dataset_train_loader),
                        sum(loss_d_list) / len(loss_d_list),
                        sum(loss_g_list) / len(loss_g_list),
                        sum(loss_g_gan_list) / len(loss_g_gan_list),
                        sum(loss_g_l1_list) / len(loss_g_l1_list),
                        sum(loss_g_hed_l1_list) / len(loss_g_hed_l1_list)))

            if iteration == img_per_epoch:
                break

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # checkpoint
        if epoch % 1 == 0:
            net_g_model_out_path = "./data/checkpoints/{}/netG_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
            net_d_model_out_path = "./data/checkpoints/{}/netD_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
            torch.save(net_g.state_dict(), net_g_model_out_path)
            torch.save(net_d.state_dict(), net_d_model_out_path)

            model_out_path = "./data/checkpoints/{}/model_epoch_{}.pth".format(run_info,
                                                                        epoch)
            torch.save({'epoch': epoch,
                        'Generator': net_g.state_dict(),
                        'Discriminator': net_d.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'scheduler_g': net_g_scheduler.state_dict(),
                        'scheduler_d': net_d_scheduler.state_dict()
                        }, model_out_path)

            log.info('Checkpoint saved to {}'.format(run_info))
        if epoch % 200 == 0:
            today = datetime.date.today()
            net_g_model_out_path = "./data/checkpoints/{}/{}_{}.pth".format(
                run_info, run_info, today.strftime("%Y%m%d"))
            torch.save(net_g.state_dict(), net_g_model_out_path)

# data_path = '/media/dong/94a07df8-6863-42d6-86f3-c96e626447dd/HE_IHC_KKM/HE_IHC_Slides/data/patch_select_0708_size25k_stratio005_ratio201_Train'
# train(run_info='model', dataset_path=data_path, img_per_epoch =10000,
#           batch_size=1, model='unet', input_nc=3, output_nc=3, ngf=64, ndf=64,
#           epoch_count=1, niter=100, niter_decay=100, lr=0.0002,
#           lr_policy='lambda', lr_decay_iters=50, beta1=0.5, cuda=True,
#               threads=4,seed=123,lamb=10,lamb_hed=0.9,hed_normalize=False)


