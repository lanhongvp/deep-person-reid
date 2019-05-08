from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
from IPython import embed
from torchreid.utils.iotools import mkdir_if_missing, write_json, read_json

class aiCity666_veri(object):
    """
        aiCity 666 and veri 776 for train

        AiCity 666 Dataset statistics:
        # vehicles: 333（train）+ 333（test）
        # images: 36953 (train) + 18290 (test) + 1052(query)
        VeRi 776 Dataset statistics:
        # vehicles: 576（train）+ 200（test）
        # images: 37778 (train) + 11579 (test) + 1678(query)
    """
    dataset_dir = 'aiCity_veri'

    def __init__(self, root='../../lan_reid/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = osp.abspath(self.dataset_dir)
        #embed()
        self.train_dir = osp.join(self.dataset_dir, 'image_train_m')
        self._check_before_run()

        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,is_train=True,relabel=True)
        # num_total_vids = num_train_vids + num_query_vids
        num_total_vids = num_train_vids
        num_total_imgs = num_train_imgs

        print("=> aiCity 666 and VeRi 776 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_vids, num_total_imgs))
        print("  ------------------------------")

        self.train = train   
        self.num_train_vids = num_train_vids
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        

    def vid2label(self,id):
        sort_id = list(set(id))
        sort_id.sort()
        num_id = len(sort_id)
        label = []
        for x in id:
            label.append(sort_id.index(x))

        return label,num_id

    def _process_dir(self,dir_path, is_train=True,relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        vid_list = []
        if is_train:
            for img_path in img_paths:
                veid = int(img_path.split('/')[-1].split('_')[0])
                vid_list.append(veid)

            vlabel, num_vids = self.vid2label(vid_list)

            dataset = []
            count = 0
            for img_path in img_paths:
                vid = vid_list[count]
                if relabel:
                    vid = vlabel[count]
                dataset.append((img_path, vid,-1))
                count = count + 1
            num_imgs = len(dataset)

        elif not is_train:
            pass
        #embed()
        return dataset,num_vids,num_imgs


class aiCity666_vp(object):
    """
        aiCity 666 and viewpoint info

        AiCity 666 Dataset statistics:
        # vehicles: 333（train）+ 333（test）
        # images: 36953 (train) + 18290 (test) + 1052(query)
    """
    dataset_dir = 'aiCity_vp_all'

    def __init__(self, root='../../lan_reid/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = osp.abspath(self.dataset_dir)
        #embed()
        self.train_dir = osp.join(self.dataset_dir, 'image_train_all')
        # self.label_dir = self.dataset_dir + 'train_label.csv'
        self._check_before_run()

        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,is_train=True,relabel=True)
        # query, num_query_vids, num_query_imgs = self._process_dir(self.query_dir,is_train=False,is_track=False)
        # gallery, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir,is_train=False,is_track=True)
        # num_total_vids = num_train_vids + num_query_vids
        num_total_vids = num_train_vids
        num_total_imgs = num_train_imgs

        print("=> aiCity 666 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        # print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
        # print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_vids, num_total_imgs))
        print("  ------------------------------")

        self.train = train       
        self.num_train_vids = num_train_vids
        self.num_train_vpids = 5

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        
    def vid2label(self,id):
        sort_id = list(set(id))
        sort_id.sort()
        num_id = len(sort_id)
        label = []
        for x in id:
            label.append(sort_id.index(x))
        return label,num_id

    def _process_dir(self,dir_path, is_train=True,relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        vid_list = []
        if is_train:
            for img_path in img_paths:
                veid = int(img_path.split('/')[-1].split('_')[0])
                vid_list.append(veid)

            vlabel, num_vids = self.vid2label(vid_list)

            dataset = []
            count = 0
            for img_path in img_paths:
                vpid = int(img_path.split('/')[-1].split('_')[1])
                vid = vid_list[count]
                if relabel:
                    vid = vlabel[count]
                dataset.append((img_path, vid,-1,vpid))
                count = count + 1
            # embed()
            num_imgs = len(dataset)

        #embed()
        return dataset,num_vids,num_imgs


class aiCity666_raw(object):
    """
        aiCity 666 

        AiCity 666 Dataset statistics:
        # vehicles: 333（train）+ 333（test）
        # images: 36953 (train) + 18290 (test) + 1052(query)
    """
    dataset_dir = 'aiCity_s'

    def __init__(self, root='../../lan_reid/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = osp.abspath(self.dataset_dir)
        #embed()
        self.train_dir = osp.join(self.dataset_dir, 'image_train_all')
        self.query_dir = osp.join(self.dataset_dir, 'image_query_all')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test_all')
        # self.label_dir = self.dataset_dir + 'train_label.csv'
        self._check_before_run()

        train, num_train_vids, num_train_imgs = self._process_dir(self.train_dir,is_train=True,relabel=True)
        query, num_query_vids, num_query_imgs = self._process_dir(self.query_dir,is_train=False,is_track=False)
        gallery, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir,is_train=False,is_track=True)
        gallery_nt, num_gallery_vids, num_gallery_imgs = self._process_dir(self.gallery_dir,is_train=False,is_track=False)
        # num_total_vids = num_train_vids + num_query_vids
        num_total_vids = num_train_vids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> aiCity 666 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_vids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.gallery_nt = gallery_nt

        self.num_train_vids = num_train_vids
        self.num_query_vids = num_query_vids
        self.num_gallery_vids = num_gallery_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def vid2label(self,id):
        sort_id = list(set(id))
        sort_id.sort()
        num_id = len(sort_id)
        label = []
        for x in id:
            label.append(sort_id.index(x))

        return label,num_id

    def _process_dir(self,dir_path, is_train=True,relabel=False,is_track=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        test_track_dir = self.dataset_dir + '/test_track.txt'

        vid_list = []
        if is_train:
            for img_path in img_paths:
                veid = int(img_path.split('/')[-1].split('_')[0])
                vid_list.append(veid)

            vlabel, num_vids = self.vid2label(vid_list)

            dataset = []
            count = 0
            for img_path in img_paths:
                vid = vid_list[count]
                if relabel:
                    vid = vlabel[count]
                dataset.append((img_path, vid,-1))
                count = count + 1
            num_imgs = len(dataset)

        elif not is_train and is_track:
            test_tracks = open(test_track_dir)
            dataset = []
            for test_track in test_tracks.readlines():
                test_track = test_track.split()
                test_track_ids = list(map(lambda track_id_:dir_path+'/'+track_id_,test_track))
                for t_id_dir in test_track_ids:
                    dataset.append(t_id_dir)
            num_vids = -1
            num_imgs = len(dataset)
            
        elif not is_train and not is_track:
            dataset = []
            count = 1
            for img_path in img_paths:
                # vid = int(img_path.strip('.jpg').split('/')[-1])
                dataset.append((img_path,count))
                count = count + 1
            num_imgs = len(dataset)
            num_vids = -1
        #embed()
        return dataset,num_vids,num_imgs
