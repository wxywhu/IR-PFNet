from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torchvision.transforms as transforms
import lmdb
from dataset import DatasetFromHdf5,DatasetTestFromHdf5,DatasetLRTestFromHdf5,DatasetFromLMDB,DatasetFromLMDB_LF

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform():
    return Compose([
        #CenterCrop(crop_size),
        #Resize(crop_size // upscale_factor),
        
        ToTensor(),
    ])
    

def target_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
    

def get_dataset(data_dir, HR_name, upscale_factor, patch_size, data_augmentation=None):
    hr_dir = join(data_dir, HR_name)
    LR_name=HR_name.replace('HR','LR')
    lr_dir = join(data_dir,  LR_name)
    LR_FCNN_name=HR_name.replace('HR','LR_FCNN')
    lr_fcnn_dir = join(data_dir, LR_FCNN_name)

    return DatasetFromFolder(hr_dir, lr_dir, lr_fcnn_dir, patch_size, upscale_factor, data_augmentation, Thrconv = False, lr_transform=lr_transform(), lr_fcnn_transform=lr_fcnn_transform(), target_transform=target_transform())
    
####################################################
def get_h5dataset(data_dir, patch_size, data_augmentation):
    return DatasetFromHdf5(data_dir, patch_size, data_augmentation, input_transform=input_transform(),
                             target_transform=target_transform())
                             
def get_LMDBdataset(data_dir, patch_size, data_augmentation):
    return DatasetFromLMDB(data_dir, patch_size, data_augmentation, input_transform=input_transform(),
                             target_transform=target_transform())     

def get_LMDBdataset_LF(data_dir, patch_size, data_augmentation):
    return DatasetFromLMDB_LF(data_dir, patch_size, data_augmentation, input_transform=input_transform(),
                             target_transform=target_transform())    

def get_testh5dataset(data_dir):
    return DatasetTestFromHdf5(data_dir, input_transform=input_transform(),
                              target_transform=target_transform())

def get_lr_test_h5dataset(data_dir):
    return DatasetLRTestFromHdf5(data_dir, input_transform=input_transform())
###################################################  
                           
def get_testdataset(data_dir, HR_name):
    hr_dir = join(data_dir, HR_name)
    LR_name=HR_name.replace('HR','LR')
    lr_dir = join(data_dir,  LR_name)
    LR_FCNN_name=HR_name.replace('HR','LR_FCNN')
    lr_fcnn_dir = join(data_dir, LR_FCNN_name)
    
    return DatasetTestFromFolder(hr_dir, lr_dir, lr_fcnn_dir,Thrconv = False, lr_transform=lr_transform(), lr_fcnn_transform=lr_fcnn_transform(), target_transform=target_transform())                             

                                                          
def get_testset(data_dir, upscale_factor):
    return DatasetTest(data_dir, upscale_factor,input_transform=input_transform())
    
    
    
    
    
    
    