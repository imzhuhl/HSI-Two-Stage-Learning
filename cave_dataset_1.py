"""
提前加好噪声，干净图像和噪声图像放在一个npz中
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class CaveDataset(Dataset):
    def __init__(self, npz_path, scene_id=-1):
        super(CaveDataset, self).__init__()
        data = np.load(npz_path)

        clean_imgs = data['clean_img']
        noise_imgs = data['noise_img']

        # nhwc -> nchw
        clean_imgs = np.transpose(clean_imgs, (0, 3, 1, 2))
        noise_imgs = np.transpose(noise_imgs, (0, 3, 1, 2))

        if scene_id == -1:
            # convert to (n*c, h, w)
            clean_imgs = np.reshape(clean_imgs, (-1, 512, 512))
            noise_imgs = np.reshape(noise_imgs, (-1, 512, 512))
        else:
            clean_imgs = clean_imgs[scene_id]
            noise_imgs = noise_imgs[scene_id]

        self.clean_imgs = clean_imgs
        self.noise_imgs = noise_imgs

        assert len(self.clean_imgs) == len(self.noise_imgs)

        # self.pre_transform = pre_transform
        # self.inputs_transform = inputs_transform
    
    def __getitem__(self, index):
        noise_img = self.noise_imgs[index]
        clean_img = self.clean_imgs[index]
        
        # preprocess
        # chw
        noise_img = noise_img[np.newaxis, :, :]
        clean_img = clean_img[np.newaxis, :, :]

        # normolization
        # noise_img = noise_img / np.max(noise_img)
        # clean_img = clean_img / np.max(clean_img)
        noise_img = torch.from_numpy(noise_img).float()
        clean_img = torch.from_numpy(clean_img).float()
        
        sample = {
            'noise_img': noise_img,
            'clean_img': clean_img
        }
        return sample
        
    def __len__(self):
        return len(self.clean_imgs)

import random

class CaveNToN(Dataset):
    """cave noise to noise dataset, random noise
    """
    def __init__(self, npz_path, scene_id=0, interval=1):
        super(CaveNToN, self).__init__()
        data = np.load(npz_path)

        noise_imgs = data['noise_img']
        noise_imgs = noise_imgs[scene_id]  # (512, 512, 31)
        self.noise_imgs = np.transpose(noise_imgs, (2, 0, 1))  # (31, 512, 512)

        # 生成样本对
        id_pair_lst = []
        for i in range(31):
            for j in range(i-interval, i+interval+1):
                if j < 0 or j > 30 or j == i:
                    continue
                id_pair_lst.append((i, j))
        self.id_pair_lst = id_pair_lst
        # print(id_pair_lst)
        # print(len(self.id_pair_lst))

    def __getitem__(self, index):
        fp_nimg = self.noise_imgs[self.id_pair_lst[index][0]]  # (512, 512)
        lp_nimg = self.noise_imgs[self.id_pair_lst[index][1]]
        fp_nimg = fp_nimg[np.newaxis, :, :]  # (1, 512, 512)
        lp_nimg = lp_nimg[np.newaxis, :, :]
        fp_nimg = torch.from_numpy(fp_nimg).float()
        lp_nimg = torch.from_numpy(lp_nimg).float()

        sample = {
            # 'first_pair_cimg': fp_img,
            'first_pair_nimg': fp_nimg,
            'last_pair_nimg': lp_nimg,
            # 'first_pair_noise': fp_noise,
        }
        return sample

    def __len__(self):
        return len(self.id_pair_lst)

