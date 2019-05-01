import random
import os
import os.path as osp

from PIL import Image
from torchvision import  transforms
from torch.autograd import Variable
import torch

from IPython import embed
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import preprocessing as pre
from torchreid.utils.re_ranking import re_ranking

def get_track_id(root_dir,is_train=False):
    dataset = 'aiCity_s'
    dataset_dir = osp.join(root_dir,dataset)
    print('Track dataset dir',dataset_dir)
    gallery_track_name = 'test_track_id.txt'
    train_track_name = 'train_track_id.txt'
    gallery_track_dir = osp.join(dataset_dir,gallery_track_name)
    train_track_dir = osp.join(dataset_dir,train_track_name)
    print('gallery_dir',gallery_track_dir)
    print('train_dir',train_track_dir)
    
    if is_train:
        pass
    elif not is_train:
        gt_ids = {}
        with open(gallery_track_dir,'r') as gallery_track_f:
            gt_idx = 0  
            for gallery_track_id in gallery_track_f.readlines():
                gallery_tid = gallery_track_id.split()
                gt_ids[gt_idx] = map(lambda gt_id:int(gt_id),gallery_tid)
                gt_idx += 1
        return gt_ids

def track_info_average(track_id,global_f):
    # track_id: dcit
    # global_f::torch.tensor
    # local_f: torch.tensor 
    global_ft = torch.zeros((len(track_id),global_f.shape[1]))
    local_ft = torch.zeros_like(global_ft)
    t_id_st = 0
    for t_idx,t_id in enumerate(track_id):
        t_id_len = len(list(track_id[t_idx]))
        t_id_end = t_id_st + t_id_len
        global_ft[t_idx] = torch.sum(global_f[t_id_st:t_id_end,:],dim=0,keepdim=True)/t_id_len
        local_ft[t_idx] = torch.sum(global_f[t_id_st:t_id_end:,t_id_len],dim=0,keepdim=True)/t_id_len
        t_id_st = t_id_end
    return global_ft

