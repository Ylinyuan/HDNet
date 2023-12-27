#!/usr/bin/python3
#coding=utf-8
import os
from tkinter import image_names
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset_yuan
from model_HDNet  import HDNet
from apex import amp
import logging as logger
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import time
import Myloss
import numpy as np
from torchvision import transforms
from metric import StructureMeasure
import cv2



TAG = "model_HDNet"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="./log/train_%s.log"%(TAG), filemode="w")

L_color = Myloss.L_color()
L_spa = Myloss.L_spa()

L_exp = Myloss.L_exp(16,0.6)
L_TV = Myloss.L_TV()


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

####### 验证 ################
def validate(net, val_loader, nums):
    net.train(False)
    running_mae = 0.0
    running_smean = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out2r, out3r, out4r, out5r, enhance_image, A, att, weighted_sum = net(image, 1)
            pred = torch.sigmoid(out2r)
            
            for j, (label, pred) in enumerate(zip(mask.detach().cpu(),pred.detach().cpu())):
                pred_idx = pred[0,:,:].numpy()
                label_idx = label[:,:].numpy()
                running_smean += StructureMeasure(pred_idx.astype(np.float32), (label_idx>=0.5).astype(np.bool))
                running_mae += np.abs(pred_idx - label_idx).mean()
                
    epoch_mae = running_mae / 2000
    epoch_smeasure = running_smean / 2000    
    net.train(True)
    return epoch_smeasure 

######### 训练 #########################
def train(Dataset, Network):
    
    ## dataset
    ## train
    cfg    = Dataset.Config(datapath='/data/Wang/lx/Enhancement/data/train_6002_day', savepath='/data/Wang/lx/Enhancement/OUR/out_model/HDNet/111', mode='train', batch=16, lr=0.01, momen=0.9, decay=5e-4, epoch=200) # lr=0.05
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=0)
    
    ## val 
    cfg_val  = Dataset.Config(datapath='/data/Wang/lx/Enhancement/Zero-DCE-master/Zero-DCE_code/data/TOTAL/2000',mode='test')
    valdata  = Dataset.Data(cfg_val)
    val_loader = DataLoader(valdata, batch_size=4, shuffle=False, num_workers=0)
    
    best_epoch = 0
    best_sm = 0.0
    
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'DCE' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}, {'params': net.DCE.parameters(), 'lr': 1e-4}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        
        

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            out2r, out3r, out4r, out5r, enhance_image, A, att, weighted_sum = net(image,epoch)
            loss_hot = structure_loss(att, mask)
            Loss_TV = 20*L_TV(A)
            loss_spa = torch.mean(L_spa(enhance_image, image))
            loss_col = 5*torch.mean(L_color(enhance_image))
            loss_exp = 10*torch.mean(L_exp(enhance_image))
            loss2r = structure_loss(out2r, mask)
            loss3r = structure_loss(out3r, mask)
            loss4r = structure_loss(out4r, mask)
            loss5r = structure_loss(out5r, mask)
            
            loss = loss_hot/4 + (Loss_TV+loss_spa+loss_col+loss_exp) + loss2r+loss3r/2+loss4r/4+loss5r/8

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
         
            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss_hot':loss_hot.item(), 'Loss_TV':Loss_TV.item(), 'loss_spa':loss_spa.item(), 'loss_col':loss_col.item(), 'loss_exp':loss_exp.item(), 'loss2r':loss2r.item(), 'loss3r':loss3r.item(), 'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
            if step%10 == 0:
                msg_train = '%s | step:%d/%d/%d | lr=%.6f| loss_hot=%.6f| Loss_TV=%.6f | loss_spa=%.6f | loss_col=%.6f | loss_exp=%.6f | loss2r=%.6f | loss3r=%.6f | loss4r=%.6f  | loss5r=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss_hot.item(), Loss_TV.item(), loss_spa.item(), loss_col.item(), loss_exp.item(), loss2r.item(), loss3r.item(), loss4r.item(), loss5r.item(), loss.item())
                msg = "step:%d/%d/%d | loss=%.6f"%(global_step, epoch+1, cfg.epoch, loss.item())
                print(msg)
                logger.info(msg_train)
        
        
        if epoch>20:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1)) 
        
        sm_now = validate(net, val_loader, 2000)
        if sm_now > best_sm:
            best_sm = sm_now
            best_epoch = epoch + 1
        print('best epoch is:%d, S:%s' % (best_epoch, best_sm))
  
        
    
if __name__=='__main__':
    train(dataset_yuan, HDNet)
