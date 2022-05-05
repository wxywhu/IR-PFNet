

from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pfnet import PFNet as Model

from utility import *

from data import get_dataset
from data import get_h5dataset,get_LMDBdataset
from dataset import DatasetFromHdf5
from torch.utils.data import DataLoader

import pdb
import socket
import time

from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch PFNet')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument("--num_features", type=int, default=64, help="n_feats, default set to 64")
parser.add_argument("--num_colors", type=int, default=1, help="n_subs, default set to 1")

parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--patch_size', type=int, default=32, help='Size of cropped  validation image')
parser.add_argument('--data_augmentation', type=bool, default=False)

parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning Rate. Default=0.001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--lamda', type=float, default=0, help='lambda to balance')

parser.add_argument('--prefix', default='IR_PFNet', help='Location to save checkpoint models')

# Dir settings
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--train_HR_dir', type=str, default='train_HR_LR_FCNN_LF')
parser.add_argument('--val_HR_dir', type=str, default='valid_HR_LR_FCNN')

parser.add_argument('--model', type=str, default='')
parser.add_argument('--pretrained_sr', default='', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)

def train(epoch):
    epoch_loss = 0
    model.train()    
    for iteration, batch in enumerate(train_data_loader):
        inputs,target= Variable(batch[0].float()), Variable(batch[1].float())
        
        if cuda:
            inputs = inputs.cuda()    #[N,an2,C,H,W]  
            target = target.cuda()
        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(inputs)
        loss1 = criterion(prediction, target)

        
        t1 = time.time()
        loss = loss1
        epoch_loss += loss.data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss1: {:.8f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_data_loader), loss1.data, (t1-t0)))
        writer.add_scalar('train_L1', loss1.data, (epoch-1)* len(train_data_loader ) +iteration)

        writer.add_image('LR_FCNN',inputs.data[4,4,:,:,:],global_step=(epoch-1)* len(train_data_loader) +iteration) 
        writer.add_image('SR',prediction[3].data[4,:,:,:],global_step=(epoch-1)* len(train_data_loader) +iteration)  
        writer.add_image('HR',target.data[4,:,:,:],global_step=(epoch-1)* len(train_data_loader) +iteration)  
        if ((epoch-1)* len(train_data_loader) +iteration)%20==0:
            epoch_psnr = validation((epoch-1)* len(train_data_loader) +iteration)
            checkpoint((epoch-1)* len(train_data_loader) +iteration)
            if epoch_psnr>= 42:
                checkpoint((epoch-1)* len(train_data_loader) +iteration)      
    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss / len(train_data_loader )))


def validation(epoch):
    avg_psnr = 0
    avg_loss =0
    test1 = nn.MSELoss()
    test2 = L1_SAM_loss()
    
    for iteration, batch in enumerate(validation_data_loader):
        with torch.no_grad():
            inputs,target= Variable(batch[0].float()), Variable(batch[1].float())
            if cuda:
                inputs = inputs.cuda()                
                target = target.cuda()

            prediction = model(inputs)
            loss1,loss2 = test2(prediction[3], target)        
            mse = test1(prediction[3], target)
            psnr = 10 * log10(1 / mse.data)
        
            del inputs
            del prediction
            del target   
            avg_psnr += psnr
            avg_loss += loss1.data
        
    writer.add_scalar('valid_L1', avg_loss/ len(validation_data_loader), epoch)
    writer.add_scalar('val_psnr', avg_psnr/ len(validation_data_loader), epoch)
    print("===> Epoch {} Avg. PSNR: {:.4f} ".format(epoch, avg_psnr/ len(validation_data_loader)))
    return avg_psnr/ len(validation_data_loader)
    
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = os.path.join(opt.model,  opt.prefix+"X{}_epoch_{}.pth".format(opt.upscale_factor,epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

                                  
torch.backends.cudnn.enabled = False                                             
opt = parser.parse_args()
cudnn.benchmark = True
                                  
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    


print('===> Loading datasets')

train_dir = os.path.join(opt.dataset, 'x'+str(opt.upscale_factor), opt.train_HR_dir)
train_set = get_LMDBdataset(train_dir, opt.patch_size, opt.data_augmentation)
print(len(train_set))
train_data_loader =  DataLoader(train_set, num_workers=opt.threads, batch_size=opt.train_batch_size, shuffle=True, drop_last= True)

val_dir = os.path.join(opt.dataset, 'x'+str(opt.upscale_factor), opt.val_HR_dir)
print(val_dir)
val_set = get_h5dataset(val_dir, opt.patch_size, opt.data_augmentation)
validation_data_loader =  DataLoader(val_set, num_workers=opt.threads, batch_size=opt.val_batch_size, shuffle=False, drop_last= True)

print('===> Building model')

model = Model(num_colors=opt.num_colors, num_features=opt.num_features)

model=model.float()
################################################kaiming initialize###############################################
#model.initialize()
criterion = L1_Average_loss()


print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    criterion = criterion.cuda()

if opt.pretrained:
    model_name = os.path.join(opt.model, opt.pretrained_sr)
    print(model_name)
    if os.path.exists(model_name):
        print('right')
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')


optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

writer = SummaryWriter()



epoch=0
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    t0 = time.time()
    train(epoch)
    t1 = time.time()
    print("===> Epoch {} Total Time: {:.4f} s ".format(epoch, (t1-t0)))
#    if epoch%10 ==0 :
#        checkpoint(epoch)
    
     ##learning rate is decayed by a factor of 100 every half of total epochs
    if (epoch+1) == 2 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0002
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr'])) 
    if (epoch+1) == 2 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001         
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))         
    if (epoch+1) == 3 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000075
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr'])) 
    if (epoch+1) == 4:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00005           
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr'])) 
    if (epoch+1) == 5 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000025
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr'])) 
    if (epoch+1) == 6 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
        
          
writer.export_scalars_to_json("./all_scalars_json")
writer.close()

