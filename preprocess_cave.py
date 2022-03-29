import os
from matplotlib import image
import numpy as np
from numpy.core.numerictypes import sctype2char
import png
import glob
import cv2
import torch
import random
import matplotlib.pyplot as plt
import imageio

from utils import calc_psnr


def feature_max_norm(data):
    data = data / np.max(data)
    return data


def gaussian(img, mean, stddev):
    h, w = img.shape
    stddev_random = stddev * np.random.rand(1)  # 范围 stddev * (0 ~ 1)
    noise = np.random.randn(h, w) * stddev_random + mean
    # noise = np.random.randn(h, w) * stddev + mean  # 固定高斯

    return img + noise


def salt_and_pepper(img, prob=0.05):
    h, w = img.shape
    noise = np.random.rand(h, w)
    salt = np.max(img)
    pepper = np.min(img)
    noise_img = img.copy()
    noise_img[noise < prob] = salt
    noise_img[noise > 1-prob] = pepper
    return noise_img


def split_data():
    test_scenes = [3, 29, 30]
    
    # 读取 npy
    root = '/home/zhuhl/clab/datasets/cave'
    data_nps = glob.glob(f'{root}/*.npy')
    data_nps.sort()
    assert len(data_nps) == 31

    train_data = np.empty((0, 512, 512, 31), dtype=np.float32)
    test_data = np.empty((0, 512, 512, 31), dtype=np.float32)

    for scene_id in range(len(data_nps)):
        data_np = np.load(data_nps[scene_id])
        data_np = data_np.astype(np.float32)
        for ch in range(31):
            data_np[:, :, ch] = feature_max_norm(data_np[:, :, ch])
        data_np = data_np[np.newaxis,:,:,:]

        if scene_id in test_scenes:
            test_data = np.concatenate((test_data, data_np), axis=0)
        else:
            train_data = np.concatenate((train_data, data_np), axis=0)

    # add noise
    ga = 75
    noise_train_data = train_data.copy()
    noise_test_data = test_data.copy()
    for i in range(noise_train_data.shape[0]):
        for ch in range(31):
            img = noise_train_data[i, :, :, ch]
            # img = salt_and_pepper(img)
            img = gaussian(img, mean=0, stddev=ga*1.0/255.0)
            noise_train_data[i, :, :, ch] = img

    for i in range(noise_test_data.shape[0]):
        for ch in range(31):
            img = noise_test_data[i, :, :, ch]
            # img = salt_and_pepper(img)
            img = gaussian(img, mean=0, stddev=ga*1.0/255.0)
            noise_test_data[i, :, :, ch] = img
        
    # 保存数据集
    # np.savez(f'./cave/fixed{ga}/train.npz', clean_img=train_data, noise_img=noise_train_data)
    # np.savez(f'./cave/fixed{ga}/test.npz', clean_img=test_data, noise_img=noise_test_data)


def get_psnr():
    noise_type = 'randga75_im'
    suffix = 'test'
    data = np.load(f'./cave/{noise_type}/{suffix}.npz')
    clean_imgs = data['clean_img']
    noise_imgs = data['noise_img']

    # log_file = f'{noise_type}_{suffix}_psnr.txt'
    # logger = open(os.path.join('./analysis', log_file), 'w+')
    # print(f'rand gaussian noise image psnr.', file=logger)
    for scene_id in range(noise_imgs.shape[0]):
        scene_psnr = 0
        for chan_id in range(31):
            ci = clean_imgs[scene_id, :, :, chan_id]
            ni = noise_imgs[scene_id, :, :, chan_id]
            scene_psnr += calc_psnr(ci, ni)
        scene_psnr /= 31
        print(f'Scene {scene_id} : {scene_psnr}')


def output_img():
    data = np.load('./cave/randga75/test.npz')
    clean_imgs = data['clean_img']
    noise_imgs = data['noise_img']

    # for scene_id in range(28):
    #     imageio.imwrite(f'./temp/s{scene_id}_clean.png', clean_imgs[scene_id, :, :, 30])
    scene_id = 0
    for i in range(25, 31):
        ci = clean_imgs[scene_id, :, :, i]
        ni = noise_imgs[scene_id, :, :, i]
        # imageio.imwrite(f'./temp/c{i}_clean.png', ci)
        imageio.imwrite(f'./temp/c{i}_noise.png', ni)
        

def main():
    # split_data()
    # get_psnr()
    output_img()


if __name__ == '__main__':
    main()
