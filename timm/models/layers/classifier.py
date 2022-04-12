""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .linear import Linear


def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc

class TwoLayerFixedClassifier(nn.Module):
    """Two layer classifier with softmax inbetween."""

    def __init__(self,num_feat,num_intermed,num_cls,use_conv=False):
        super(TwoLayerFixedClassifier, self).__init__()
        self.num_intermed = num_intermed
        self.fc1 = _create_fc(num_feat, num_intermed, use_conv=use_conv)
        self.fc2 = _create_fc(num_intermed, num_cls, use_conv=use_conv)
        
        
    def forward(self,x):
        x = self.fc1(x)
        y = x.max(dim=1,keepdim=True)

        x = torch.zeros((x.shape[0],self.num_intermed), device=torch.device('cuda'))
        x[list(range(x.shape[0])),y[1][:,0]] = 1.0

        x = self.fc2(x)
        return x        
    

def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False, num_classes_intermed=None, usemax=False):
    global_pool, num_pooled_features = _create_pool(num_features, num_classes, pool_type, use_conv=use_conv)
    if num_classes_intermed==None:
        fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    else:
        if usemax:
            fc = TwoLayerFixedClassifier(num_pooled_features, num_classes_intermed, num_classes, use_conv)
        else:
            fc1 = _create_fc(num_pooled_features, num_classes_intermed, use_conv=use_conv)
            fc2 = _create_fc(num_classes_intermed, num_classes, use_conv=use_conv)
            fc = nn.Sequential(fc1,fc2)
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0., use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.fc(x)
            return self.flatten(x)
