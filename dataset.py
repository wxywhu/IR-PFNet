import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image
import random
from random import randrange
from scipy.io import loadmat
import lmdb
import h5py
from torchvision.transforms import Compose, ToTensor

def input_transform():
    return Compose([
        #CenterCrop(crop_size),
        #Resize(crop_size // upscale_factor),
        ToTensor(),    #(C,H, W)Tensor��ʽ��normalization  ( /255)   to [0,1.0]
    ])

def target_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
    

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def is_h5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])

def is_lmdb_file(filename):
    return any(filename.endswith(extension) for extension in [".mdb"])
    
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img

def load_HRmatdata(filepath):
    data = loadmat(filepath)['HR']
    return data

def load_LRmatdata(filepath):
    data = loadmat(filepath)['LR']
    return data
    
def toTensor(input):
    data = torch.from_numpy(input.copy()).float()
    return data
    
def turn_to_4D(data):
    data_new = data.reshape([1, data.shape[0], data.shape[1], data.shape[2]])
    return data_new
    
def get_patch(img_ins,target, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_ins[0].shape
    
    ip = patch_size
    
    if ix == -1:
        ix = random.randrange(0, iw - patch_size + 1)
    if iy == -1:
        iy = random.randrange(0, ih - patch_size + 1)

    img_ins_out = []

    for i in range(len(img_ins)):
        img_in = img_ins[i]
        img_in = img_in[ iy:iy + ip, ix:ix + ip] 
        img_in = np.expand_dims(img_in, axis=2)
        img_ins_out.append(img_in)
        

    target = target[ iy:iy + ip, ix:ix + ip]    
    target = np.expand_dims(target, axis=2)
    

    return img_ins_out, target
    
def get_patch_test(img_ins,target):
    img_ins_out = []

    for i in range(len(img_ins)):
        img_in = img_ins[i]
        img_in = img_in[:,:]
        img_in = np.expand_dims(img_in, axis=2)
        img_ins_out.append(img_in)
        
    target = target[:,:]    
    target = np.expand_dims(target, axis=2)
    

    return img_ins_out, target

def get_patch_test_LF(img_ins, targets):

    img_ins_out = []
    targets_out = []
    for i in range(len(img_ins)):
        img_in = img_ins[i]
        img_in = img_in[:,:]
        img_in = np.expand_dims(img_in, axis=2)
        img_ins_out.append(img_in)

    for i in range(len(targets)):
        target = targets[i]
        target = target[:,:]
        target = np.expand_dims(target, axis=2)
        targets_out.append(target)

    return img_ins_out, targets_out


def get_patch_lr_test(img_in0,img_in1,img_in2,img_in3):
     
    img_in0 = img_in0[:, :]    
    img_in0 = np.expand_dims(img_in0, axis=2)
    
    img_in1 = img_in1[:, :]    
    img_in1 = np.expand_dims(img_in1, axis=2)
    
    img_in2 = img_in2[:, :]   
    img_in2 = np.expand_dims(img_in2, axis=2)
    
    img_in3 = img_in3[:, :]    
    img_in3 = np.expand_dims(img_in3, axis=2)
    

    return img_in0, img_in1, img_in2, img_in3



def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = torch.from_numpy(np.flip(img_in,axis=2).copy())
        img_tar = torch.from_numpy(np.flip(img_tar,axis=2).copy())
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = torch.from_numpy(np.flip(img_in,axis=1).copy())
            img_tar = torch.from_numpy(np.flip(img_in,axis=1).copy())
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in =  torch.from_numpy(np.ascontiguousarray(np.transpose(img_in,(0,2,1))[np.newaxis,:]))
            img_tar = torch.from_numpy(np.ascontiguousarray(np.transpose(img_tar,(0,2,1))[np.newaxis,:]))
            info_aug['trans'] = True

    return img_in, img_tar, info_aug



class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation, Thrconv = False, input_transform=None,target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_mat_file(x)]
        self.lr_dir = LR_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        #self.dataset = dataset
        self.data_augmentation = data_augmentation
        self.Thrconv = Thrconv
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        target = load_HRmatdata(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        target_name = os.path.splitext(file)[0]
        input_name=target_name.replace('HR','LR')
        
        input = load_LRmatdata(os.path.join(self.lr_dir,input_name+'.mat')) 
                                                            
        input, target = get_patch(input, target, self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)
                              
        if self.Thrconv:
            input, target= turn_to_4D(input,target) 
                              
        if self.input_transform:
            input = self.input_transform(input)
                              
        if self.target_transform:
            target = self.target_transform(target)
                                                   
        return input, target

    def __len__(self):
        return len(self.image_filenames)
    
class DatasetFromLMDB(data.Dataset):
    def __init__(self, file_path, patch_size, data_augmentation=False,input_transform=None,target_transform=None):
        super(DatasetFromLMDB, self).__init__()
        #self.opt = opt
        self.filenames = file_path
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.env = lmdb.open(self.filenames)
        self.txn = self.env.begin(write=False) 
                
    def __getitem__(self, index):
    #    print(index)
        data0_bin = self.txn.get((str(index)+'LR0').encode())
    #    print(type(data0_bin))
        data0_buf = np.frombuffer(data0_bin, dtype="uint8")
        data0 = data0_buf.reshape(self.patch_size, self.patch_size)
        
        data1_bin = self.txn.get((str(index)+'LR1').encode())
        data1_buf = np.frombuffer(data1_bin, dtype="uint8")
        data1 = data1_buf.reshape(self.patch_size, self.patch_size)
        
        data2_bin = self.txn.get((str(index)+'LR2').encode())
        data2_buf = np.frombuffer(data2_bin, dtype="uint8")
        data2 = data2_buf.reshape(self.patch_size, self.patch_size)
        
        data3_bin = self.txn.get((str(index)+'LR3').encode())
        data3_buf = np.frombuffer(data3_bin, dtype="uint8")
        data3 = data3_buf.reshape(self.patch_size, self.patch_size)

        data4_bin = self.txn.get((str(index)+'LR4').encode())
        data4_buf = np.frombuffer(data4_bin, dtype="uint8")
        data4 = data4_buf.reshape(self.patch_size, self.patch_size)

        data5_bin = self.txn.get((str(index)+'LR5').encode())
        data5_buf = np.frombuffer(data5_bin, dtype="uint8")
        data5 = data5_buf.reshape(self.patch_size, self.patch_size)

        data6_bin = self.txn.get((str(index)+'LR6').encode())
        data6_buf = np.frombuffer(data6_bin, dtype="uint8")
        data6 = data6_buf.reshape(self.patch_size, self.patch_size)

        data7_bin = self.txn.get((str(index)+'LR7').encode())
        data7_buf = np.frombuffer(data7_bin, dtype="uint8")
        data7 = data7_buf.reshape(self.patch_size, self.patch_size)

        data8_bin = self.txn.get((str(index)+'LR8').encode())
        data8_buf = np.frombuffer(data8_bin, dtype="uint8")
        data8 = data8_buf.reshape(self.patch_size, self.patch_size)
        
        ref_ind = 4
        if(ref_ind==0):
            target_bin = self.txn.get((str(index)+'HR0').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size)
        if(ref_ind==1):
            target_bin = self.txn.get((str(index)+'HR1').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size)           
        if(ref_ind==2):
            target_bin = self.txn.get((str(index)+'HR2').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size)   
        if(ref_ind==3):
            target_bin = self.txn.get((str(index)+'HR3').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size) 
        if(ref_ind==4):
            target_bin = self.txn.get((str(index)+'HR4').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size) 
        if(ref_ind==5):
            target_bin = self.txn.get((str(index)+'HR5').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size) 
        if(ref_ind==6):
            target_bin = self.txn.get((str(index)+'HR6').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size) 
        if(ref_ind==7):
            target_bin = self.txn.get((str(index)+'HR7').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size)   
        if(ref_ind==8):
            target_bin = self.txn.get((str(index)+'HR8').encode())
            target_buf = np.frombuffer(target_bin, dtype="uint8")
            target = target_buf.reshape(self.patch_size, self.patch_size) 

        inputs = [data0,data1,data2,data3,data4,data5,data6,data7,data8]  #list
        inputs,target = get_patch_test(inputs,target)    #list[numpy]  
        
       # print(input0.shape)
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        
        inputs_tensor = []
        if self.input_transform:
            for i in range(len(inputs)):
                input = self.input_transform(inputs[i])
                inputs_tensor.append(input)           ##list[tensor]
     
        inputs_last = torch.cat(inputs_tensor,0).unsqueeze(1)     #[an2,c,h,w]
        
       # print(input0.shape)                      
        if self.target_transform:
            target = self.target_transform(target)           
           
        return inputs_last, target 
    
    def __len__(self):
        num = self.txn.stat()['entries'] // 18   
    #    num = 146944    
        return num

class DatasetFromLMDB_LF(data.Dataset):
    def __init__(self, file_path, patch_size, data_augmentation=False,input_transform=None,target_transform=None):
        super(DatasetFromLMDB_LF, self).__init__()
        #self.opt = opt
        self.filenames = file_path
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.env = lmdb.open(self.filenames)
        self.txn = self.env.begin(write=False) 
                
    def __getitem__(self, index):
        
        data0_bin = self.txn.get((str(index)+'LR0').encode())
        data0_buf = np.frombuffer(data0_bin, dtype="uint8")
        data0 = data0_buf.reshape(self.patch_size, self.patch_size)
        
        data1_bin = self.txn.get((str(index)+'LR1').encode())
        data1_buf = np.frombuffer(data1_bin, dtype="uint8")
        data1 = data1_buf.reshape(self.patch_size, self.patch_size)
        
        data2_bin = self.txn.get((str(index)+'LR2').encode())
        data2_buf = np.frombuffer(data2_bin, dtype="uint8")
        data2 = data2_buf.reshape(self.patch_size, self.patch_size)
        
        data3_bin = self.txn.get((str(index)+'LR3').encode())
        data3_buf = np.frombuffer(data3_bin, dtype="uint8")
        data3 = data3_buf.reshape(self.patch_size, self.patch_size)

        data4_bin = self.txn.get((str(index)+'LR4').encode())
        data4_buf = np.frombuffer(data4_bin, dtype="uint8")
        data4 = data4_buf.reshape(self.patch_size, self.patch_size)

        data5_bin = self.txn.get((str(index)+'LR5').encode())
        data5_buf = np.frombuffer(data5_bin, dtype="uint8")
        data5 = data5_buf.reshape(self.patch_size, self.patch_size)

        data6_bin = self.txn.get((str(index)+'LR6').encode())
        data6_buf = np.frombuffer(data6_bin, dtype="uint8")
        data6 = data6_buf.reshape(self.patch_size, self.patch_size)

        data7_bin = self.txn.get((str(index)+'LR7').encode())
        data7_buf = np.frombuffer(data7_bin, dtype="uint8")
        data7 = data7_buf.reshape(self.patch_size, self.patch_size)

        data8_bin = self.txn.get((str(index)+'LR8').encode())
        data8_buf = np.frombuffer(data8_bin, dtype="uint8")
        data8 = data8_buf.reshape(self.patch_size, self.patch_size)
        

        target0_bin = self.txn.get((str(index)+'HR0').encode())
        target0_buf = np.frombuffer(target0_bin, dtype="uint8")
        target0 = target0_buf.reshape(self.patch_size, self.patch_size)

        target1_bin = self.txn.get((str(index)+'HR1').encode())
        target1_buf = np.frombuffer(target1_bin, dtype="uint8")
        target1 = target1_buf.reshape(self.patch_size, self.patch_size)           

        target2_bin = self.txn.get((str(index)+'HR2').encode())
        target2_buf = np.frombuffer(target2_bin, dtype="uint8")
        target2 = target2_buf.reshape(self.patch_size, self.patch_size)   
 
        target3_bin = self.txn.get((str(index)+'HR3').encode())
        target3_buf = np.frombuffer(target3_bin, dtype="uint8")
        target3 = target3_buf.reshape(self.patch_size, self.patch_size)

        target4_bin = self.txn.get((str(index)+'HR4').encode())
        target4_buf = np.frombuffer(target4_bin, dtype="uint8")
        target4 = target4_buf.reshape(self.patch_size, self.patch_size)

        target5_bin = self.txn.get((str(index)+'HR5').encode())
        target5_buf = np.frombuffer(target5_bin, dtype="uint8")
        target5 = target5_buf.reshape(self.patch_size, self.patch_size)           

        target6_bin = self.txn.get((str(index)+'HR6').encode())
        target6_buf = np.frombuffer(target6_bin, dtype="uint8")
        target6 = target6_buf.reshape(self.patch_size, self.patch_size)   
 
        target7_bin = self.txn.get((str(index)+'HR7').encode())
        target7_buf = np.frombuffer(target7_bin, dtype="uint8")
        target7 = target7_buf.reshape(self.patch_size, self.patch_size)  

        target8_bin = self.txn.get((str(index)+'HR8').encode())
        target8_buf = np.frombuffer(target8_bin, dtype="uint8")
        target8 = target8_buf.reshape(self.patch_size, self.patch_size)  

        inputs = [data0,data1,data2,data3,data4,data5,data6,data7,data8]   #list
        targets = [target0,target1,target2,target3,target4,target5,target6,target7,target8]  #list

        inputs,targets = get_patch_test_LF(inputs,targets)    #list[numpy]  
        
       # print(input0.shape)
        if self.data_augmentation:
            input, target, _ = augment(inputs, target)
        
        inputs_tensor = []
        if self.input_transform:
            for i in range(len(inputs)):
                input = self.input_transform(inputs[i])
                inputs_tensor.append(input)           ##list[tensor]
     
        inputs_last = torch.cat(inputs_tensor,0).unsqueeze(1)     #[an2,c,h,w]
        
        targets_tensor = []                   
        if self.target_transform:
            for i in range(len(targets)):
                target = self.target_transform(targets[i])
                targets_tensor.append(target)

        targets_last = torch.cat(targets_tensor,0).unsqueeze(1)        
           
        return inputs_last, targets_last


    def __len__(self):
        num = self.txn.stat()['entries'] //8
    #    num = 146944    
        return num

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, patch_size, data_augmentation=False,input_transform=None,target_transform=None):
                             
        super(DatasetFromHdf5, self).__init__()
        self.filenames = [join(file_path, x) for x in listdir(file_path) if is_h5_file(x)]
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.input_transform = input_transform
        self.target_transform = target_transform
   
    def __getitem__(self, index):
        hf = h5py.File(self.filenames[index])
        
        data0 = hf.get('data0')
        data1 = hf.get('data1')
        data2 = hf.get('data2')
        data3 = hf.get('data3')
        data4 = hf.get('data4')
        data5 = hf.get('data5')
        data6 = hf.get('data6')
        data7 = hf.get('data7')
        data8 = hf.get('data8')
        target = hf.get('label') 

        inputs = [data0,data1,data2,data3,data4,data5,data6,data7,data8]  #list
        inputs,target = get_patch(inputs,target,self.patch_size)    #list[numpy]  
        
       # print(input0.shape)
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        
        inputs_tensor = []
        if self.input_transform:
            for i in range(len(inputs)):
                input = self.input_transform(inputs[i])
                inputs_tensor.append(input)           ##list[tensor]
     
        inputs_last = torch.cat(inputs_tensor,0).unsqueeze(1)

        if self.target_transform:
            target = self.target_transform(target)           
           
        return inputs_last, target
    
    def __len__(self):
        return len(self.filenames)
        

class DatasetTestFromHdf5(data.Dataset):
    def __init__(self, file_path, input_transform=None, target_transform=None):
                             
        super(DatasetTestFromHdf5, self).__init__()
        self.filenames = sorted([join(file_path, x) for x in listdir(file_path) if is_h5_file(x)])
        self.input_transform = input_transform
        self.target_transform = target_transform
   
    def __getitem__(self, index):
        hf = h5py.File(self.filenames[index])
        
        data0 = hf.get('data0')
        data1 = hf.get('data1')
        data2 = hf.get('data2')
        data3 = hf.get('data3')
        data4 = hf.get('data4')
        data5 = hf.get('data5')
        data6 = hf.get('data6')
        data7 = hf.get('data7')
        data8 = hf.get('data8')
        
        target = hf.get('label') 
        
        input_dir = self.filenames[index]     
        _, file = os.path.split(input_dir) 

        inputs = [data0,data1,data2,data3,data4,data5,data6,data7,data8]  #list

        inputs,target = get_patch_test(inputs,target)    #list[numpy]
    
        inputs_tensor = []
        if self.input_transform:
            for i in range(len(inputs)):
                input = self.input_transform(inputs[i])
                inputs_tensor.append(input)           ##list[tensor]
     
        inputs_last = torch.cat(inputs_tensor,0).unsqueeze(1)     #[an2,c,h,w]
    #    print(inputs_last.size()) 

        if self.target_transform:
            target = self.target_transform(target)   
           
        return  inputs_last, target, file
    
    def __len__(self):
        return len(self.filenames)
        
        
class DatasetLRTestFromHdf5(data.Dataset):
    def __init__(self, file_path, input_transform=None):
                             
        super(DatasetLRTestFromHdf5, self).__init__()
        self.filenames = [join(file_path, x) for x in listdir(file_path) if is_h5_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
   
    def __getitem__(self, index):
        hf = h5py.File(self.filenames[index])
        
        data0 = hf.get('data0')
        data1 = hf.get('data1')
        data2 = hf.get('data2')
        data3 = hf.get('data3')
        
        input_dir = self.filenames[index]     
        _, file = os.path.split(input_dir)  
      
        input0,input1,input2,input3= get_patch_lr_test(data0,data1,data2,data3)
        
        if self.input_transform:
            input0 = self.input_transform(input0)            
            input1 = self.input_transform(input1)
            input2 = self.input_transform(input2)
            input3 = self.input_transform(input3)
                                       
        return input0, input1, input2, input3, file
    
    def __len__(self):
        return len(self.filenames)       

class DatasetTest(data.Dataset):
    def __init__(self, file_path, upscale_factor, data_augmentation=False,input_transform=None):
                             
        super(DatasetTest, self).__init__()
        self.filenames = [join(file_path, x) for x in listdir(file_path) if is_mat_file(x)]
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation
        self.input_transform = input_transform
   
    def __getitem__(self, index):
        input = load_LRmatdata(self.filenames[index])
        input_dir = self.filenames[index]     
        _, file = os.path.split(input_dir)
        
#        if self.data_augmentation:
#            input, target, _ = augment(input, target)
        
        if self.input_transform:
            input = self.input_transform(input)
                                                 
        return input, file
    
    def __len__(self):
        return len(self.filenames)
