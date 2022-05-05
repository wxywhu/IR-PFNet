
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
from math import sqrt
import math
import torch.nn.functional as F
from dcn.deform_conv import ModulatedDeformConv


# Residual block
class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x
    

# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, size_k, size_p):
        super(dense_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, growthRate,
                                 kernel_size=size_k, stride=1, padding=size_p, bias=False)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.act(self.conv(x)) 
        out = torch.cat((x, out), 1)
        return out

# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, size_k, size_p, num_layer=3):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, size_k, size_p))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = nn.Conv2d(in_channels_, in_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out
    
class FE(nn.Module):
    def __init__(self, num_colors, num_features):
        super(FE, self).__init__()

        self.conv0 = nn.Conv2d(num_colors, num_features,
                                 kernel_size=5, stride=1, padding=2, bias=False) 
        self.rb = nn.Sequential(RB(num_features),
                                RB(num_features),
                                RB(num_features))
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        outs = []
        for i in range(3):
            out = self.act(self.conv0(x[i]))
            out = self.rb(out)
            outs.append(out)
            
        return outs

class PPDFM(nn.Module):
    def __init__(self, num_colors, num_features):
        super(PPDFM, self).__init__()

        # feature 
        self.ppdfb =  PPDFB(num_colors, num_features)
        self.pdfb = PDFB(num_colors, num_features)
                	  
    def forward(self, x):   
        ########input is a list####### #[[N,nf,H,W] ,[N,nf,H,W],[N,nf,H,W]]
        for i in range(3):
            x = self.ppdfb(x)
            
        out = self.pdfb(x)       
        return out
       
class PPDFB(nn.Module):
    def __init__(self, num_colors, num_features):
        super(PPDFB, self).__init__()

        # feature 
        self.conv0 =  nn.Conv2d(num_features, num_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)   
        self.pdfb = PDFB(num_colors, num_features)
                		                       
        self.merge = nn.Conv2d(2*num_features, num_features,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.rdb =  RDB(num_features,num_features,3,1)                           
        self.act = nn.LeakyReLU(0.2)
        	  
    def forward(self, x):   
        ########input is a list####### #[[N,nf,H,W] ,[N,nf,H,W],[N,nf,H,W]]
        spa_feats = [] 
               
        for i in range (3):
            spa_feat = self.act(self.conv0(x[i]))
            spa_feats.append(spa_feat)
            
        fused_feat = self.pdfb(spa_feats)
            
        out_feats = []
        for i in range (3):
            res_feat = self.act(self.merge(torch.cat((fused_feat,spa_feats[i]),1)))
            res_feat = self.rdb(res_feat)
            out_feat =  res_feat + x[i]                     
            out_feats.append(out_feat)

        return out_feats

class PDFB(nn.Module):
    def __init__(self, num_colors, num_features, deform_ks=3):
        super(PDFB, self).__init__()
        self.num_colors = num_colors
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.num_features  = num_features
    
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # feature   
        self.conv_l1 = nn.Conv2d(num_features, num_colors,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv_l2_1 = nn.Conv2d(num_colors, num_colors,
                                 kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_l2_2 = nn.Conv2d(num_colors, num_colors,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv_l3_1 = nn.Conv2d(num_colors, num_colors,
                                 kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_l3_2 = nn.Conv2d(num_colors, num_colors,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        
        # off_mask
        self.cat_off = nn.Conv2d(3*num_colors, num_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
                                                                 							 
        self.rb_off = nn.Sequential(RB(num_features),
                                RB(num_features),
                                RB(num_features))
         
        self.offset_mask = nn.Conv2d(num_features, 3*self.num_colors*3*self.size_dk, 
                                 kernel_size=3, stride=1, padding=1)
        self.deform_conv = ModulatedDeformConv(
            3*self.num_colors, num_features, deform_ks, padding=deform_ks//2, deformable_groups=3*self.num_colors
            )   
        self.off_msk_conv = nn.Conv2d(2* 3*self.num_colors*3*self.size_dk, 3*self.num_colors*3*self.size_dk,
                                 kernel_size=3, stride=1, padding=1)
        
        self.fused_feat_conv =  nn.Conv2d(2* num_features, num_features, 
                                 kernel_size=3, stride=1, padding=1)
        		                                                  
        self.act = nn.LeakyReLU(0.2)
        	  
    def forward(self, x):   
        ########input is a list####### #[[N,nf,H,W] ,[N,nf,H,W],[N,nf,H,W]]
        in_feats_l1 = []
        in_feats_l2 = []
        in_feats_l3 = []
        for i in range(3):
            #L1
            in_feat_l1 = self.act(self.conv_l1(x[i])) #[N,1,H,W]
            in_feats_l1.append(in_feat_l1)
            #L2
            in_feat_l2 = self.act(self.conv_l2_1(in_feat_l1))
            in_feat_l2 = self.act(self.conv_l2_2(in_feat_l2))
            in_feats_l2.append(in_feat_l2)
            #L3
            in_feat_l3 = self.act(self.conv_l3_1(in_feat_l2))
            in_feat_l3 = self.act(self.conv_l3_2(in_feat_l3))
            in_feats_l3.append(in_feat_l3)
            
        cat_feat_l1 = torch.cat(in_feats_l1,1) ##[N,3,H,W]
        cat_feat_l2 = torch.cat(in_feats_l2,1) ##[N,3,H/2,W/2]
        cat_feat_l3 = torch.cat(in_feats_l3,1) ##[N,3,H/4,W/4]
        cat_feats = [cat_feat_l1, cat_feat_l2, cat_feat_l3]
       
        fused_feats = []
        for i in range(3, 0, -1):
            off_feat = self.cat_off(cat_feats[i-1])            
            off_feat = self.rb_off(off_feat)          
            off_msk = self.offset_mask(off_feat)
            off = off_msk[:, :3*self.num_colors*2*self.size_dk, ...]
            msk = torch.sigmoid(
                off_msk[:, 3*self.num_colors*2*self.size_dk:, ...]
            )
            if i < 3:
                off_msk = self.off_msk_conv(torch.cat([off_msk,upsampled_off_mask],1))
                off = off_msk[:, :3*self.num_colors*2*self.size_dk, ...]
                msk = torch.sigmoid(
                    off_msk[:, 3*self.num_colors*2*self.size_dk:, ...]
                )
            
            fused_feat = self.act(self.deform_conv(cat_feats[i-1], off, msk))  ##[N,nf,H,W]

            if i < 3:
                fused_feat = self.fused_feat_conv(torch.cat([fused_feat,upsampled_fused_feat],dim=1))
                
            if i > 1:
                upsampled_off_mask = self.up2(off_msk) * 2
                upsampled_fused_feat = self.up2(fused_feat)
        
        return fused_feat
    
class IGF(nn.Module):
    def __init__(self, num_colors, num_features, num_group=4):
        super(IGF, self).__init__()
        self.num_colors = num_colors
        self.num_group = num_group
        self.num_features  = num_features
        self.convs = nn.ModuleDict()
        self.masks = nn.ModuleDict()
        for i in range(2):
            level = f'l{i}'
            self.convs[level] = nn.Conv2d(num_features, num_features,
                                 kernel_size=3, stride=1, padding=1, bias=False) 
            self.masks[level] = nn.Conv2d(num_features, 1,
                                 kernel_size=3, stride=1, padding=1, bias=False) 
            
        self.merge_conv = nn.Conv2d(2*num_features, num_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        
        self.rdb_out = nn.Sequential(RDB(num_features,num_features,3,1),
                                    RDB(num_features,num_features,3,1),
                                    RDB(num_features,num_features,3,1),
                                    RDB(num_features,num_features,3,1))  
        self.out = nn.Conv2d(num_features, num_features, 
                                 kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2)
        	  
    def forward(self, x):  
        ########input is a list####### 
        fused_feats = []
        fused_masks = []
        for i in range(2):
            level = f'l{i}'
            fused_feat =self.act( self.convs[level](x[i]))      ##N,nf,H,W
            fused_feats.append(fused_feat.unsqueeze(2)) ###N,nf,1,H,W
            fused_mask = self.masks[level](fused_feat) ##N,1,H,W
            fused_masks.append(fused_mask.unsqueeze(2))  ##N,1,1,H,W
            
        merged_fused_feat =  torch.cat(fused_feats,2) ##N,nf,G,H,W
        merged_fused_mask =  torch.cat(fused_masks,2) ##N,1,G,H,W
        
        merged_fused_mask =  F.softmax(merged_fused_mask, dim =2)   ##N,1,G,H,W
        merged_feat = torch.mul(merged_fused_mask, merged_fused_feat)  ##N,nf,G,H,W
        merged_feat_out = torch.cat((merged_feat[:,:,0,:,:],merged_feat[:,:,1,:,:]),1)   ##N,nf*G,H,W
        
        merged_out0 = self.act(self.merge_conv(merged_feat_out))
        merged_out1 = self.rdb_out(merged_out0)
        res = self.act(self.out(merged_out1))	
        
        return res
    
class PFNet(nn.Module):
    def __init__(self, num_colors, num_features, num_group=4, deform_ks=3):
        super(PFNet, self).__init__()
        self.num_colors = num_colors
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.num_group = num_group
        self.num_features  = num_features

        self.feat_ext = FE(num_colors, num_features)  
        self.ppdfm_g1 =  PPDFM(num_colors, num_features)  
        self.ppdfm_g2 =  PPDFM(num_colors, num_features) 
        self.ppdfm_g3 =  PPDFM(num_colors, num_features) 
        self.ppdfm_g4 =  PPDFM(num_colors, num_features) 
        self.merge = IGF(num_colors, num_features)
        self.out = nn.Conv2d(num_features, 1, 
                                 kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2)
        			  
    def forward(self, x):   
        
        N,an2,C,H,W = x.shape#[N,an2,C,H,W]
        x_center = x[:,4,:,:,:]
        input_g1 = [ x[:, 3, :, :, :], x[:, 4, :, :, :], x[:, 5, :, :, :] ]  #0
        input_g2 = [ x[:, 1, :, :, :], x[:, 4, :, :, :], x[:, 7, :, :, :] ]  #90
        input_g3 = [ x[:, 0, :, :, :], x[:, 4, :, :, :], x[:, 8, :, :, :] ]  #45
        input_g4 = [ x[:, 2, :, :, :], x[:, 4, :, :, :], x[:, 6, :, :, :] ]  #135
        
        feat_g1 = self.feat_ext(input_g1)
        feat_g2 = self.feat_ext(input_g2)
        feat_g3 = self.feat_ext(input_g3)
        feat_g4 = self.feat_ext(input_g4)
        ###  extrat pyramid inputs   #[N,an*C,H,W] 
        
        fused_feat_g1 = self.ppdfm_g1(feat_g1)
        fused_feat_g2 = self.ppdfm_g2(feat_g2)
        fused_feat_g3 = self.ppdfm_g3(feat_g3)
        fused_feat_g4 = self.ppdfm_g4(feat_g4)
    
        
        fused_feats1 = [fused_feat_g1,fused_feat_g1]
        merged_res1 = self.merge(fused_feats1)
        res1 = self.act(self.out(merged_res1))
        out1 = torch.add(res1, x_center)
        
        fused_feats2 = [merged_res1,fused_feat_g2]
        merged_res2 = self.merge(fused_feats2)
        res2 = self.act(self.out(merged_res2))
        out2 = torch.add(res2, x_center)
        
        fused_feats3 = [merged_res2,fused_feat_g3]
        merged_res3 = self.merge(fused_feats3)
        res3 = self.act(self.out(merged_res3))
        out3 = torch.add(res3, x_center)
        
        fused_feats4 = [merged_res3,fused_feat_g4]
        merged_res4 = self.merge(fused_feats4)
        res4 = self.act(self.out(merged_res4))
        out4 = torch.add(res4, x_center)
        
        out_average = (out1 + out2 + out3 + out4)/4
        out = [out1, out2, out3, out4, out_average]	
        
        return out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight,a=0.2, mode='fan_in', nonlinearity='leaky_relu')          
                  