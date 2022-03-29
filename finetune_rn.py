import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dncnn_net import DnCNN
import cave_dataset_1
from utils import get_criterion, calc_psnr, save_train, init_exps
from zcfg import FT_CFG

CUDA_ID = 7
DEVICE = torch.device(f'cuda:{CUDA_ID}')


class FT1:
    def __init__(self) -> None:
        ft_cfg = FT_CFG['randga25_a0.0']
        self.alpha = ft_cfg['alpha']
        
        # init experiments
        base_name = f"ft_s{ft_cfg['scene_id']}_itv{ft_cfg['interval']}"
        # base_name = f"ft_s{ft_cfg['scene_id']}_wo_itv{ft_cfg['interval']}"
        log_dir = init_exps(os.path.join(ft_cfg['log_dir'], base_name))
        ft_cfg['log_dir'] = log_dir
        writer = SummaryWriter(log_dir)

        # record parameter
        with open(os.path.join(log_dir, 'params.yaml'), 'w') as f:
            yaml.dump(ft_cfg, f, default_flow_style=False, allow_unicode=True)
        
        logger = open(os.path.join(log_dir,'logger.txt'),'w+')
        pretrain_path = ft_cfg['pretrain_path']

        # build model
        model = DnCNN(channels=1, num_of_layers=ft_cfg['num_of_layers'])
        model.load_state_dict(torch.load(pretrain_path, map_location='cpu')['model'], strict=True)
        model = model.to(DEVICE)

        # load dataset
        print('Loading dataset...')
        finetune_dataset_path, test_dataset_path = ft_cfg['finetune_dataset_path'], ft_cfg['test_dataset_path']
        finetune_dataset = cave_dataset_1.CaveNToN(finetune_dataset_path, interval=ft_cfg['interval'])
        finetune_loader = DataLoader(finetune_dataset, batch_size=ft_cfg['batch_size'], shuffle=True)
        test_dataset = cave_dataset_1.CaveDataset(test_dataset_path, ft_cfg['scene_id'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 不打乱按顺序

        # train
        self.finetuning(model, ft_cfg, finetune_loader, test_loader, logger, writer)

    def finetuning(self, model, args, finetune_loader, test_loader, logger, writer):
        """"""
        model_save_dir = os.path.join(args['log_dir'], 'models')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        
        # set optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
        criterion = get_criterion(losses_types=['l1', 'l2'], factors=[self.alpha, 1-self.alpha])
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 200], gamma=0.1, last_epoch=-1)

        print('Start training...', file=logger, flush=True)
        test_bar = tqdm(total=len(test_loader), bar_format="{l_bar}{bar:30}{r_bar}")
        ft_bar = tqdm(total=len(finetune_loader), bar_format="{l_bar}{bar:30}{r_bar}")
        best_psnr = 0.0
        for epoch in range(args['epoch']):
            print('Epoch number {}'.format(epoch), file=logger, flush=True)
            ft_bar.set_description(f"[{epoch}/{args['epoch']-1}]")
            test_bar.set_description(f"[{epoch}/{args['epoch']-1}]")
            
            model.train()
            finetune_loss = 0.0
            for i, sample in enumerate(finetune_loader):
                fp_nimg = sample['first_pair_nimg'].to(DEVICE)
                lp_nimg = sample['last_pair_nimg'].to(DEVICE)
                # forward & backward
                optimizer.zero_grad()
                noise_res = model(fp_nimg)
                denoise_img = fp_nimg - noise_res
                loss = criterion(denoise_img, lp_nimg)
                loss.backward()
                optimizer.step()

                # record training info
                finetune_loss += loss.item()

                ft_bar.update(1)
            ft_bar.reset()
            # scheduler.step()

            finetune_loss /= len(finetune_loader)
            # make sure epoch 0 is pretrained model test result
            scene_psnr = self.valid(model, test_loader, test_bar)
            print('[{}/{}] | finetune_loss: {:.5f} | scene_psnr: {:.4f}'
                    .format(epoch, args['epoch']-1, finetune_loss, scene_psnr), file=logger, flush=True)
            
            writer.add_scalar('Loss/fintune', finetune_loss, epoch)
            writer.add_scalar('PSNR/scene', scene_psnr, epoch)

            # save every epoch
            if best_psnr < scene_psnr:
                best_psnr = scene_psnr
                save_train(model_save_dir, model, optimizer, epoch=epoch)

        
    def valid(self, model, data_loader, pbar):
        model.eval()
        test_psnr = 0.0
        scene_psnr = 0.0
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                noise_img = sample['noise_img'].to(DEVICE)
                clean_img = sample['clean_img']

                noise_res = model(noise_img)
                denoise_img = noise_img - noise_res
                
                denoise_img = denoise_img.detach().cpu().numpy()
                clean_img = clean_img.numpy()
                psnr = calc_psnr(clean_img, denoise_img)
                scene_psnr += psnr

                pbar.update(1)
            pbar.reset()

        return scene_psnr / 31


if __name__ == '__main__':
    FT1()
