from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from IPython import embed

__all__ = ['DenseNet121']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class DenseNet121(nn.Module):
    def __init__(self, num_classes_vid, neck,neck_feat,**kwargs):
        super(DenseNet121, self).__init__()
        # self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes_vid
        self.neck = neck
        self.neck_feat = neck_feat
        
        if self.neck=='no':
            self.classifier = nn.Linear(1024, num_classes_vid)
#        self.classifier_vpid = nn.Linear(1024, num_classes_vpid)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(1024)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(1024,self.num_classes,bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        self.feat_dim = 1024

    def forward(self, x):
        x = self.base(x)
        # global_feat = F.avg_pool2d(x, x.size()[2:])
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.size(0), -1)
        # embed()
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            # embed()
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat
#         if not self.training:
#             return f
#         y_vid = self.classifier(f)
# #        y_vpid = self.classifier_vpid(f)

#         if self.loss == {'xent'}:
#             return y_vid,
#         elif self.loss == {'xent', 'htri'}:
#             return y_vid, f
#         else:
#             raise KeyError("Unsupported loss: {}".format(self.loss))
