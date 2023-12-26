#!/usr/bin/python3
#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset_yuan
from model_HDNet  import HDNet


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot=r'E:\sod\111vxiu\model-best', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out2r, out3r, out4r, out5r, enhance_image, r, x4_fam = self.net(image)
                out = out2r

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()
    
    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                out2r, out3r, out4r, out5r, enhance_image, r, x4_fam = self.net(image, shape)
                out   = out2r
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                pred = (torch.sigmoid(out[0])*255).permute(1, 2, 0).cpu().numpy()
                pred = pred[..., ::-1] 
                head  = r'E:\sod\111vxiu'
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':
    for path in [r'E:\sod\ceshiji']:
        t = Test(dataset_yuan, HDNet, path)
        print(111111)
        t.save()
        # t.show()
