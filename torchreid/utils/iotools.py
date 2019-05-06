from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
import shutil

import torch


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best=False, fpath='checkpoint.pth.tar'):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

def dict_key_slice(ori_dict, start, end):
    """
    dict slice according to the key
    :param ori_dict: original dict
    :param start: start idx
    :param end: end idx
    :return: slice dict
    """
    if end != -1:
        slice_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[start:end]}
    else:
        slice_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[start:]}
    return slice_dict


def dict_value_slice(ori_dict,st,ed):
    """
    dict slice according to the value
    the original dict value could be sliced
    :param ori_dict: original dict
    :param st: start idx
    :param ed: end idx
    :return: slice dict
    """
    slice_dict = {}
    for item in ori_dict.items():
        vid = item[0]
        vnames = item[1]
        tmp_name = []
        if ed==-1:
            for i in range(st,len(vnames)):
                tmp_name.append(vnames[i])
        else:
            for i in range(st,ed):
                tmp_name.append(vnames[i])
        slice_dict[vid] = tmp_name
    return slice_dict


def write_pickle_aicity(download_path):
    # write the aicity dataset to pickle file with vehicle id and viewpoint
    # 1: front 2: front-side 3:side 4:rear-side 5:rear
    if not os.path.isdir(download_path):
        print('please change the download_path')

    train_label = download_path + '/train_label.csv'
    train_label_vp = download_path + '/labels.txt'
    tnames = {}
    tnames_p = open('tnames_aic.pkl','wb')

    # for root, dirs, files in os.walk(train_path, topdown=True):
    with open(train_label,'r') as f1,open(train_label_vp,'r') as f2:
        for line1 in f1.readlines():
            tname = line1.strip('\n').split(',')
            vid = int(tname[0])
            timg = [tname[1]] 
            tnames[timg] = vid
        idx = 1
        for line2 in f2.readlines():
            timg = str(idx).zfill(6)+'.jpg'
            vp = line2
            tnames[timg] += '_{}'.format(vp)
            idx += 1
        pickle.dump(tnames,tnames_p)
        tnames_p.close()
        f.close()

def write_pickle_veri(download_path):
    if not os.path.isdir(download_path):
        print('please change the download_path')

    train_label = download_path + '/name_train.txt'
    tnames = {}
    tnames_p = open('tnames_veri.pkl','wb')

    # for root, dirs, files in os.walk(train_path, topdown=True):
    with open(train_label,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            tname = line.strip('\n').split('_')
            vid = int(tname[0])
            timg = [line] 
            if vid in tnames:
                tnames[vid] += timg
            else:
                tnames[vid] = timg
        pickle.dump(tnames,tnames_p)
        tnames_p.close()
        f.close()

def copy_ori2dst(ori_dict,ori_path,save_path):
    """
    copy ori folder to destination folder
    :param ori_dict: original dict
    :param ori_path: the original path 
    :param save_path: the final path which is going to be saved
    :return: none
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for item in ori_dict.items():
        tvid_tvp = item[1]
        tvid = tvid_tvp.split('_')[0]
        tvp = tvid_tvp.split('_')[1]
        timgs = item[0]
        # print(timgs)
        for timg in timgs:
            src_path = ori_path + '/' + timg
            dst_path = save_path
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path+'/'+tvid+'_'+tvp+'_'+timg)


def ori2dst_split(ori_dict,ori_path,save_path):
    """
    copy ori folder to destination folder
    :param ori_dict: original dict
    :param ori_path: the original path 
    :param save_path: the final path which is going to be saved
    :return: none
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for item in ori_dict.items():
        tvid = item[0]
        timgs = item[1]
        # print(timgs)
        for timg in timgs:
            src_path = ori_path + '/' + timg
            dst_path = save_path
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/'+tvid+'_'+timg)