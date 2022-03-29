import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dncnn_net import DnCNN
import cave_dataset_1
from utils import weights_init_kaiming, get_criterion, calc_psnr, save_train, init_exps
from zcfg import TRAIN_CFG

CUDA_ID = 7
DEVICE = torch.device(f'cuda:{CUDA_ID}')


class Trainer:
    def __init__(self):
        train_cfg = TRAIN_CFG['randga25_a0.0']
        self.alpha = train_cfg['alpha']
        # init experiments
        base_name = f"pretrained"
        log_dir = init_exps(os.path.join(train_cfg['log_dir'], base_name))
        train_cfg['log_dir'] = log_dir
        writer = SummaryWriter(log_dir)

        # record parameter
        with open(os.path.join(log_dir, 'params.yaml'), 'w') as f:
            yaml.dump(train_cfg, f, default_flow_style=False, allow_unicode=True)

        logger = open(os.path.join(log_dir,'logger.txt'),'w+')

        # build model
        # pretrain_path = train_cfg['pretrain_path'] if train_cfg['pretrained'] else None
        model = DnCNN(channels=1, num_of_layers=train_cfg['num_of_layers'])
        # if train_cfg['pretrained']:
        #     model.load_state_dict(torch.load(pretrain_path, map_location='cpu')['model'], strict=True)
        # else:
        model.apply(weights_init_kaiming)
        model = model.to(DEVICE)

        # train
        self.train(model, train_cfg, logger, writer)

    def train(self, model, args, logger, writer):
        # load dataset
        train_loader, val_loader = self.get_train_val_loaders(
            args['train_dataset_path'], args['val_dataset_path'], args['batch_size'])

        # set optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
        criterion = get_criterion(losses_types=['l1', 'l2'], factors=[self.alpha, 1-self.alpha])
        # criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 100], gamma=0.1, last_epoch=-1)

        model_save_dir = os.path.join(args['log_dir'], 'models')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        print('Start training...', file=logger, flush=True)
        train_bar = tqdm(total=len(train_loader), bar_format="{l_bar}{bar:30}{r_bar}")
        val_bar = tqdm(total=len(val_loader), bar_format="{l_bar}{bar:30}{r_bar}")
        for epoch in range(args['epoch']):
            print('Epoch number {}'.format(epoch), file=logger, flush=True)
            train_bar.set_description(f"[{epoch}/{args['epoch']-1}]")
            val_bar.set_description(f"[{epoch}/{args['epoch']-1}]")
            
            model.train()
            train_loss = 0.0
            train_psnr = 0.0
            for i, sample in enumerate(train_loader):
                noise_img = sample['noise_img'].to(DEVICE)
                clean_img = sample['clean_img'].to(DEVICE)

                # forward & backward
                optimizer.zero_grad()
                noise_res = model(noise_img)
                denoise_img = noise_img - noise_res
                loss = criterion(denoise_img, clean_img)
                loss.backward()
                optimizer.step()

                # record training info
                train_loss += loss.item()
                denoise_img = denoise_img.detach().cpu().numpy()
                denoise_img = np.clip(denoise_img, 0., 1.)
                clean_img = clean_img.detach().cpu().numpy()
                train_psnr += calc_psnr(clean_img, denoise_img)

                train_bar.update(1)
            train_bar.reset()
            scheduler.step()

            train_psnr /= len(train_loader)
            train_loss /= len(train_loader)
            val_loss, val_psnr = self.valid(model, val_loader, val_bar)
            print('[{}/{}] | train_loss: {:.5f} | train_psnr: {:.3f} | val loss: {:.5f} | val_psnr: {:.3f}'
                    .format(epoch, args['epoch']-1, train_loss, train_psnr, val_loss, val_psnr), file=logger, flush=True)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('PSNR/train', train_psnr, epoch)
            writer.add_scalar('PSNR/val', val_psnr, epoch)

            # save every epoch
            save_train(model_save_dir, model, optimizer, epoch=epoch)

    def valid(self, model, data_loader, val_bar):
        criterion = get_criterion(losses_types=['l1', 'l2'], factors=[self.alpha, 1-self.alpha])
        # criterion = nn.MSELoss()
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                noise_img = sample['noise_img'].to(DEVICE)
                clean_img = sample['clean_img'].to(DEVICE)

                noise_res = model(noise_img)
                denoise_img = noise_img - noise_res
                loss = criterion(denoise_img, clean_img)
                
                val_loss += loss.item()
                denoise_img = denoise_img.detach().cpu().numpy()
                denoise_img = np.clip(denoise_img, 0., 1.)
                clean_img = clean_img.detach().cpu().numpy()
                val_psnr += calc_psnr(clean_img, denoise_img)

                val_bar.update(1)
            val_bar.reset()

        return val_loss / len(data_loader), val_psnr / len(data_loader)
    
    def get_train_val_loaders(self, train_dataset_path, val_dataset_path, batch_size):
        print('Loading dataset...')
        train_dataset = cave_dataset_1.CaveDataset(train_dataset_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = cave_dataset_1.CaveDataset(val_dataset_path)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader


def main():
    Trainer()


if __name__ == '__main__':
    main()
