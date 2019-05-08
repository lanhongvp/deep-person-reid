from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['DenseNet121']


class DenseNet121(nn.Module):
    def __init__(self, num_classes_vid, loss={'xent'}, **kwargs):
        super(DenseNet121, self).__init__()
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier_vid = nn.Linear(1024, num_classes_vid)
        self.classifier_vpid = nn.Linear(1024, num_classes_vpid)
        self.feat_dim = 1024

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y_vid = self.classifier_vid(f)
        y_vpid = self.classifier_vpid(f)

        if self.loss == {'xent'}:
            return y_vid, y_vpid
        elif self.loss == {'xent', 'htri'}:
            return y_vid, y_vpid, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
