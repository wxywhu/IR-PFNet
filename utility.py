
import os
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
def gauss(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8
    
    s = sigma**2
    sum_val =  0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += kernel[i, j]
    
    kernel = kernel/sum_val
    
    return kernel


class GaussianBlurConv(nn.Module):
    def __init__(self,  in_channels, kernel_size, sigma, padding):
        super(GaussianBlurConv, self).__init__()
        self.channels = in_channels
        self.padding = padding
        kernel = gauss(kernel_size, sigma)  
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        return x
        

                
class L1_SAM_loss(nn.Module):
    """L1_SAM_loss."""
    def __init__(self):
        super(L1_SAM_loss, self).__init__()
        self.eps = 1e-7
                  
    def forward(self, X, Y):
                       
        loss1=torch.mean(torch.abs(torch.add(X,-Y)))
                 
        tem1 = torch.sum(X*Y, dim=-3)  
                    
        tem2 =torch.norm(X, dim=-3)*torch.norm(Y, dim=-3)
        
        tem3 = tem1/(tem2+self.eps)
        
        tem3 = torch.clamp(tem3,-1,1)  
                       
        loss2 =torch.mean(torch.acos(tem3))/np.pi
                       
        return loss1, loss2

class L1_Average_loss(nn.Module):
    """L1_SAM_loss."""
    def __init__(self):
        super(L1_Average_loss, self).__init__()
        self.eps = 1e-7
                  
    def forward(self, X, Y):
                       
        loss1=torch.mean(torch.abs(torch.add(X[0],-Y)))
        loss2=torch.mean(torch.abs(torch.add(X[1],-Y)))
        loss3=torch.mean(torch.abs(torch.add(X[2],-Y)))
        loss4=torch.mean(torch.abs(torch.add(X[3],-Y)))
            
        loss =  (loss1 + loss2 + loss3 + loss4)/4
                 
                       
        return loss
    
    
class L2_SAM_loss(nn.Module):
    """L2_SAM_loss."""
    def __init__(self):
        super(L2_SAM_loss, self).__init__()
        self.esp = 1e-12
                  
    def forward(self, X, Y):
                       
        loss1= torch.mean(func.mse_loss(X, Y, sum))/2
        nom=torch.mul(X,Y).sum(dim=1) 
        denominator = torch.norm(X, p=2, dim=1, keepdim=True).clamp(min=self.esp)*torch.norm(Y, p=2, dim=1, keepdim=True).clamp(min=self.esp) 
        denominator = denominator.squeeze()                      
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0
        loss2 = torch.mean(sam) 
                      
        return loss1, loss2
        
class L2_D_Loss(nn.Module):
    def __init__(self,sequence_len):
        super(L2_D_Loss, self).__init__()
        self.sequence_len = sequence_len
        
    def forward(self, X, Y):
        loss3 = 0
        for i in range(self.sequence_len): 
            if i<= self.sequence_len -2:            
                difference_p = X[:,i+1,:,:]-X[:,i,:,:] 
                difference_t = Y[:,i+1,:,:]-Y[:,i,:,:]            
                loss3_mse= torch.mean(func.mse_loss(difference_p, difference_t, sum))/2
                loss3+= loss3_mse/(self.sequence_len-1)
            return loss3
            










            
def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)     
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)     
    
    

