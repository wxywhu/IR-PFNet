from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from os import listdir
from scipy.misc import imsave
import scipy.io as sio
import time
from os.path import join
from scipy.io import loadmat
from scipy.io import savemat

import imageio 

from data import get_testdataset,get_testh5dataset,get_lr_test_h5dataset
from utility import *

from pfnet import PFNet as Model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch PFNet')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument("--num_features", type=int, default=64, help="n_feats, default set to 64")
parser.add_argument("--num_colors", type=int, default=1, help="n_subs, default set to 1")

parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')

parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--gpu_num', default=1, type=int, help='number of gpu')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--test_dir', type=str, default='')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='test_HR_LR_FCNNclass/')

parser.add_argument('--model', default='', help='sr pretrained base model')

parser.add_argument('--dataset_name', default='', help='The name of dataset')
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)


torch.backends.cudnn.enabled = False                                             
opt = parser.parse_args()
cudnn.benchmark = True

print(opt)
t0 = time.time()
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    
#print('===> Building model')
#model= Net(opt.testBatchSize, opt.sequence_length, 0, opt.upscale_factor, opt.hidden_size, test=True, test_h, test_w)
#
#if cuda:
#    model = model.cuda()
#    model = torch.nn.DataParallel(model)
#
#if os.path.exists(opt.model):
#  model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
#  print('Pre-trained SR model is loaded.')



def eval():
  print('===> Building model')
  model = Model(num_colors=opt.num_colors, num_features=opt.num_features)
  model=model.float()   
  if os.path.exists(opt.model):
    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    else:
        state_dict = torch.load(opt.model, map_location=lambda storage, loc: storage)
# create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
# load params
        model.load_state_dict(new_state_dict)
    print('Pre-trained SR model is loaded.')

        
    avg_psnr = 0
    avg_ssim = 0
    test1 = nn.MSELoss()
    ssim_loss = SSIM()    
    for iteration, batch in enumerate(test_dataloader , 1):
        with torch.no_grad():
            inputs,target,file= Variable(batch[0].float()), Variable(batch[1].float()), batch[2]

            name = file[0].split('_')
            print(name[0])
            if cuda:

                inputs = inputs.cuda()
                target = target.cuda()
            
            t0 = time.time()
            if opt.chop_forward:
                prediction = chop_forward(inputs, model, opt.upscale_factor)
            else:
                prediction = model(inputs)
                prediction = prediction[3]
            t1 = time.time()
            print("Timer: {:.4f} sec".format(t1-t0))
        
            save_png(prediction.data, name[0])
            t1 = time.time()
            print("Timer: {:.4f} sec".format(t1-t0))   
            mse = test1(prediction[:,:,:,:], target[:,:,:,:])
            psnr = 10 * log10(1 / mse.data)
            ssim = ssim_loss( prediction,target)        
            del target
            del prediction   
            avg_psnr += psnr
            avg_ssim += ssim
            print("===> PSNR: {:.8f} dB ||SSIM: {:.8f}".format(psnr,ssim))
    print(len(test_dataloader))
    print("===> PSNR: {:.8f} dB ||SSIM: {:.8f}".format(avg_psnr / len(test_dataloader),avg_ssim / len(test_dataloader)))		        
            
def save_png(img, img_name):
    img_name= img_name.replace('.h5', '_SR.png')
    print(img_name)
    img=img.cpu()
    data = img.squeeze().clamp(0,1).numpy()  #[H,W]
    data = (data * 255).astype(np.uint8)
    save_dir=os.path.join(opt.output, opt.dataset_name,'x'+ str(opt.upscale_factor))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = os.path.join(save_dir, img_name)
    imageio.imsave(save_fn,data)
    

#####x8_forward and chop_forward taken from https://github.com/thstkdgus35/EDSR-PyTorch
#####EDSR and MDSR Team from SNU
def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        return Variable(ret, volatile=v.volatile)

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output
    

def chop_forward(x, model, scale, shave=16):
    b, num, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:,:,:, 0:h_size, 0:w_size],
        x[:,:,:, 0:h_size, (w - w_size):w],
        x[:,:,:, (h - h_size):h, 0:w_size],
        x[:,:,:, (h - h_size):h, (w - w_size):w]] 
    outputlist = []
    for i in range(4):
      input_batch = inputlist[i]
      output_batch = model(input_batch)
      outputlist.append(output_batch)
    
    output = Variable(x.data.new(b, c, h, w))
    print(output.shape)
    output[:,:,:, 0:h_half, 0:w_half] = outputlist[0][:, :, :, 0:h_half, 0:w_half]
    print(output.shape)     
    output[:,:,:, 0:h_half, w_half:w] = outputlist[1][:, :, :, 0:h_half, (w_size - w + w_half):w_size]
    print(output.shape)    
    output[:,:,:, h_half:h, 0:w_half] = outputlist[2][:, :, :, (h_size - h + h_half):h_size, 0:w_half]
    print(output.shape)    
    output[:,:,:, h_half:h, w_half:w] = outputlist[3][:, :, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    print(output.shape)
    
    return output
#['air-conditionings', 'bicycles', 'buildings', 'cars', 'peoples', 'others']
###Eval Start!!!!
#eval()
file_name = ['buildings', 'cars', 'peoples' ,'others']
for i in range(len(file_name)):
    print('===> Loading datasets')
    test_dir = join(opt.test_dir,'x'+str(opt.upscale_factor),opt.test_dataset, file_name[i] )
    testset = get_testh5dataset(test_dir)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    eval()
    print(file_name[i]+ 'eval end!')
    print('*****************************************************************************')
    print('*****************************************************************************')
