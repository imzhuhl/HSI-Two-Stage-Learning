import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
from torch import optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import scipy.io
import time

from dncnn_net import DnCNN
import cave_dataset_1
from utils import weights_init_kaiming, get_criterion, calc_psnr, save_train, init_exps
from zcfg import TEST_CFG


CUDA_ID = 1
DEVICE = torch.device(f'cuda:{CUDA_ID}')


class Tester:
    def __init__(self) -> None:
        test_cfg = TEST_CFG['randga75_im_a0.8_pretrained']
        self.scene_id = test_cfg['scene_id']

        saved_model_path = test_cfg['model_path']
        test_data_path = test_cfg['test_dataset_path']
        result_dir = test_cfg['result_dir']
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        logger = open(os.path.join(result_dir, 'test_log.txt'), 'w+')

        # load pretrained model
        print(f'Load weights: {saved_model_path}')
        model = DnCNN(channels=1, num_of_layers=test_cfg['num_of_layers'])
        model.load_state_dict(torch.load(saved_model_path, map_location='cpu')['model'], strict=True)
        model = model.to(DEVICE)

        # load test dataset
        test_dataset = cave_dataset_1.CaveDataset(test_data_path, scene_id=self.scene_id)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # inference
        start = time.time()
        result_imgs, our_psnr_list, noise_psnr_list = self.test(model, test_loader)
        end = time.time()
        print(f"exec time: {end - start}")

        # show results
        img_saved_dir = os.path.join(result_dir, 'images')
        if not os.path.exists(img_saved_dir):
            os.mkdir(img_saved_dir)
        self.show_results(result_imgs, our_psnr_list, noise_psnr_list, img_saved_dir, logger)
        # self.save_to_mat(result_imgs, result_dir)

    def test(self, model, test_loader):
        model.eval()
        our_index_list = []
        noise_index_list = []
        result_imgs = []

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader), bar_format="{l_bar}{bar:30}{r_bar}")
            for i, sample in enumerate(test_loader):
                # if i < self.scene_id * 31 or i >= (self.scene_id+1) * 31:
                #     continue
                noise_img, clean_img = sample['noise_img'].to(DEVICE), sample['clean_img'].to(DEVICE)
                pred_noise = model(noise_img)
                denoise_img = noise_img - pred_noise

                denoise_img = denoise_img.detach().cpu().numpy()
                # denoise_img = np.clip(denoise_img, 0, 1)
                clean_img = clean_img.detach().cpu().numpy()
                noise_img = noise_img.detach().cpu().numpy()
                # noise_img = np.clip(noise_img, 0, 1)

                clean_img = np.squeeze(clean_img)
                denoise_img = np.squeeze(denoise_img)
                noise_img = np.squeeze(noise_img)

                our_psnr = calc_psnr(clean_img, denoise_img)
                noise_psnr = calc_psnr(clean_img, noise_img)
                our_psnr = calc_psnr(clean_img, denoise_img)
                noise_psnr = calc_psnr(clean_img, noise_img)
                our_ssim = ssim(clean_img, denoise_img, data_range=1.0)
                noise_ssim = ssim(clean_img, noise_img, data_range=1.0)
                our_index_list.append({'psnr': our_psnr, 'ssim': our_ssim})
                noise_index_list.append({'psnr': noise_psnr, 'ssim': noise_ssim})

                result_imgs.append({
                    'clean_img': clean_img, 
                    'noise_img': noise_img, 
                    'denoise_img': denoise_img})
                pbar.update(1)
            pbar.close()

        return result_imgs, our_index_list, noise_index_list

    def show_results(self, result_imgs, our_index_list, noise_index_list, img_saved_dir, logger):
        """show psnr and result images"""
        our_psnr_list = [item['psnr'] for item in our_index_list]
        our_ssim_list = [item['ssim'] for item in our_index_list]
        noise_psnr_list = [item['psnr'] for item in noise_index_list]
        noise_ssim_list = [item['ssim'] for item in noise_index_list]
        our_psnr_list = np.array(our_psnr_list)
        noise_psnr_list = np.array(noise_psnr_list)
        our_ssim_list = np.array(our_ssim_list)
        noise_ssim_list = np.array(noise_ssim_list)

        table = PrettyTable(['method','psnr','ssim'])

        # total average psnr
        avg_our_psnr = np.mean(our_psnr_list)
        avg_our_ssim = np.mean(our_ssim_list)
        avg_noise_psnr = np.mean(noise_psnr_list)
        avg_noise_ssim = np.mean(noise_ssim_list)
        table.add_row(['noise', f'{avg_noise_psnr:.3f}', f'{avg_noise_ssim:.4f}'])
        table.add_row(['our', f'{avg_our_psnr:.3f}', f'{avg_our_ssim:.4f}'])
        print(table, file=logger)

        # scene and channel psnr
        table = PrettyTable(['band', 'noise psnr', 'noise ssim', 'our psnr', 'our ssim'])
        for i in range(len(our_psnr_list)):
            table.add_row([i, f'{noise_psnr_list[i]:.3f}', f'{noise_ssim_list[i]:.4f}', f'{our_psnr_list[i]:.3f}', f'{our_ssim_list[i]:.4f}'])        
        print(table, file=logger)

    def plot_result(self, clean_img, noise_img, denoise_img, name, saved_dir):
        """plot clean/noisy/orig images"""

        clean_img = np.squeeze(clean_img)
        noise_img = np.squeeze(noise_img)
        denoise_img = np.squeeze(denoise_img)

        # clean_img, noise_img, denoise_img = hyper2im(clean_img), hyper2im(noise_img), hyper2im(denoise_img)

        sub_typ = 221

        plt.subplot(sub_typ)
        plt.imshow(clean_img, cmap='gray')
        plt.title('original')
        plt.gca().axis('off')
        
        plt.subplot(sub_typ + 1)
        plt.imshow(noise_img, cmap='gray')
        plt.title('noise {:.2f} db'.format(calc_psnr(clean_img, noise_img)))
        plt.gca().axis('off')

        plt.subplot(sub_typ + 2)
        plt.imshow(denoise_img, cmap='gray')
        plt.title('our {:.2f} db'.format(calc_psnr(clean_img, denoise_img)))
        plt.gca().axis('off')

        plt.savefig(os.path.join(saved_dir, f'{name}.png'))
        plt.clf()

        noise_img = np.clip(noise_img, 0, 1)
        imageio.imwrite(os.path.join(saved_dir, f'{name}_orig.png'), clean_img)
        imageio.imwrite(os.path.join(saved_dir, f'{name}_noise.png'), noise_img)
        imageio.imwrite(os.path.join(saved_dir, f'{name}_denoise.png'), denoise_img)

    def save_to_mat(self, result_imgs, saved_dir):
        clean_imgs = [item['clean_img'] for item in result_imgs]
        # noise_imgs = [item['noise_img'] for item in result_imgs]
        denoise_imgs = [item['denoise_img'] for item in result_imgs]

        clean_imgs = np.array(clean_imgs, dtype=np.float32)
        # noise_imgs = np.array(noise_imgs, dtype=np.float32)
        denoise_imgs = np.array(denoise_imgs, dtype=np.float32)
        clean_imgs = np.transpose(clean_imgs, (1, 2, 0))
        # noise_imgs = np.transpose(noise_imgs, (1, 2, 0))
        denoise_imgs = np.transpose(denoise_imgs, (1, 2, 0))

        path = os.path.join(saved_dir, 'hsits.mat')
        scipy.io.savemat(path, {'img': clean_imgs, 'img_n': denoise_imgs})


def main():
    Tester()


if __name__ == '__main__':
    main()
