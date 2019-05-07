from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchreid import data_manager
from torchreid.dataset_loader import ImageDataset,ImageDatasetNo,ImageDataset_vp
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.test_CMC import track_info_average,get_track_id
from torchreid.eval_metrics import evaluate_aicity,eval_aicity_track
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optim
from IPython import embed


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='aicity_raw',
                    choices=data_manager.get_names())
parser.add_argument('-d_m', '--dataset_m', type=str, default='aicity666_vp',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--lambda-xent', type=float, default=1,
                    help="weight to balance cross entropy loss")
parser.add_argument('--lambda-htri', type=float, default=1,
                    help="weight to balance hard triplet loss")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")
# AiCity test setting
parser.add_argument('--exp',type=str,default='exp0',help='aicity txt result name')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')
parser.add_argument('--use_track_info',action='store_true',help='whether to use track info')
parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    
    dataset_m = data_manager.init_imgreid_dataset(
        root=args.root, name=args.dataset_m, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    print("Dataset merge {}".format(dataset_m))
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset_vp(dataset_m.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset_m.train, args.train_batch, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )


    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes_vid=dataset_m.num_train_vids, num_classes_vpid=dataset_m.num_train_vpids,loss={'xent'})
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.label_smooth:
        criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset_m.num_train_vids, use_gpu=use_gpu)
    else:
        criterion_xent = nn.CrossEntropyLoss()
    criterion_htri = TripletLoss(margin=args.margin)
    
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        if check_isfile(args.load_weights):
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            # rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            # print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        if args.use_track_info:
            g_track_id = get_track_id(args.root)
            distmat = test(model, queryloader, galleryloader, use_gpu, 
                           dataset_q=dataset.query, dataset_g=dataset.gallery, 
                           track_id=g_track_id, return_distmat=True)    
        else:
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)
        if args.vis_ranked_res:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(args.save_dir, 'ranked_results'),
                topk=20,
            )
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()
        
        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            # print("==> Test")
            # rank1 = test(model, queryloader, galleryloader, use_gpu)
            # is_best = rank1 > best_rank1
            
            # if is_best:
            #     best_rank1 = rank1
            #     best_epoch = epoch + 1
            is_best = True

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    # print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, vids, _, vpids) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        if use_gpu:
            imgs, vids, vpids = imgs.cuda(), vids.cuda(), vpids.cuda()
        
        outputs_vid,outputs_vpid = model(imgs)
        if args.htri_only:
            if isinstance(features, tuple):
                loss = DeepSupervision(criterion_htri, features, vids)
            else:
                loss = criterion_htri(features, vids)
        else:
            if isinstance(outputs_vid, tuple):
                xent_loss = DeepSupervision(criterion_xent, outputs_vid, vids)
            else:
                # xent_loss_vid = criterion_xent(outputs_vid, vids)
                xent_loss_vpid = criterion_xent(outputs_vpid, vpids)
                xent_loss = xent_loss_vpid
            
            
            loss = args.lambda_xent * xent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), vids.size(0))
        xent_losses.update(xent_loss.item(),vids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Xent Loss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, xent_loss=xent_losses))
        
        end = time.time()


if __name__ == '__main__':
    main()
