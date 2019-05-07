from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
import shutil
from shutil import copyfile
import pickle
import torch
from IPython import embed

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

def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y)
    return z

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


def make_list(rootpath, savename):
    imgs = os.listdir(rootpath)
    results = []
    for img in imgs:
        results.append(os.path.join(rootpath, img) + '\n')
    results.sort()
    with open(savename, 'w') as f:
        f.writelines(results)

def concate_label_vp(rootpath):
    com_label = rootpath + '/labels_c.txt'
    com_img = rootpath + '/imglist_c.txt'
    part_label = rootpath + '/labels_p.txt'
    part_img = rootpath + '/imglist_p.txt'
    whole_vp_label = {}
    tnames_p = open('tnames_aic.pkl','wb')
    com_vp_label = combine_label_img(com_img,com_label)
    part_vp_label = combine_label_img(part_img,part_label)
    print('common keys',com_vp_label.keys() & part_vp_label.keys())
#    print('com keys',com_vp_label.keys())
#    print('part keys',part_vp_label.keys())    
    whole_vp_label = merge_two_dicts(com_vp_label,part_vp_label)
    pickle.dump(whole_vp_label,tnames_p)
    tnames_p.close()

def combine_label_img(img_dir,label_dir):
    # label_dir: viewpoint label dir
    # img_dir: img dir
    # return: dict with key-img_name value-viewpoint and vehicle id label
    cnt = 0
    with open(label_dir,'r') as f:
        for index, line in enumerate(f):
            cnt += 1
    print('img_dir {} cnt {}'.format(img_dir,cnt))
    f1 = open(img_dir,'r')
    f2 = open(label_dir,'r')
    vp_label = {}
    for i in range(cnt):
        img_path = f1.readline()
        # print(img_path)
        img_name = img_path.strip('\n').split('/')[-1]
        # img_name = img_name.split('\\')[-1]
        # print('img_name\n',img_name)
        vid = img_name.split('_')[0]
        # print('vid\n',vid)
        # img_name = img_name.split('_')[1]
        # print('img_name\n',img_name)
        vpid = f2.readline()
        vid_vpid = vid+'_'+vpid
        vp_label[img_name] = vid_vpid
    f1.close()
    f2.close()
    return vp_label

def write_pickle_aicity(download_path):
    # write the aicity dataset to pickle file with vehicle id and viewpoint
    # 1: front 2: front-side 3:side 4:rear-side 5:rear
    if not os.path.isdir(download_path):
        print('please change the download_path')

    train_label = download_path + '/imglist.txt'
    train_label_vp = download_path + '/labels.txt'
    tnames = {}
    tnames_p = open('tnames_aic.pkl','wb')
    # get the total count lines
    count = 0
    with open(train_label,'r') as f:
        for index, line in enumerate(f):
            count += 1
    print('train num',count)

    f1 = open(train_label,'r')
    f2 = open(train_label_vp,'r')
    # for root, dirs, files in os.walk(train_path, topdown=True):
    for i in range(count):
        img_path = f1.readline()
        # print(img_path)
        img_name = img_path.strip('\n').split('/')[-1]
        # img_name = img_name.split('\\')[-1]
        # print('img_name\n',img_name)
        vid = img_name.split('_')[0]
        # print('vid\n',vid)
        # img_name = img_name.split('_')[1]
        # print('img_name\n',img_name)
        vpid = f2.readline()
        vid_vpid = vid+'_'+vpid
        tnames[img_name] = vid_vpid
    pickle.dump(tnames,tnames_p)
    tnames_p.close()
    f1.close()
    f2.close()

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
    print('Total cnt',len(ori_dict.items()))
    for item in ori_dict.items():
        tvid_tvp = item[1].strip('\n')
        if len(tvid_tvp)==1:
            break
        # print('tvid tvp',tvid_tvp)
        tvid = tvid_tvp.split('_')[0]
        tvp = tvid_tvp.split('_')[1]
        # embed()
        timgs = item[0]
        # print(timgs)
        # for timg in timgs:
        src_path = ori_path + '/' + timgs
        dst_path = save_path
        # embed()
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path+'/'+tvid+'_'+tvp+'_'+timgs.split('_')[1])
        # embed()

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

