""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), acc_pm1=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    scores, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    # count class id +/1 as correct
    if acc_pm1:
        tr = target.reshape(1, -1).expand_as(pred)
        correct = torch.where(torch.abs(pred.sub(tr)) <2,1,0)
    else:
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
   
    
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    
def prec_rec_per_class(output, target, num_classes, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    
    apr = {}
    
    for k in topk:
        apr[k] = {}
    
    
    for c in range(num_classes):
                
        cc = target.reshape(1, -1).expand_as(pred)==c
        ntrue = cc[:1].reshape(-1).float().sum(0)
        
        correct_c = correct.logical_and(cc)

        
        for k in topk:
            
            pred_c = pred[:min(k, maxk)].reshape(-1)==c
            npred = pred_c.float().sum(0) 
            
            ncorrect = correct_c[:min(k, maxk)].reshape(-1).float().sum(0)

            
            apr[k][c] = [ (batch_size-(ntrue-ncorrect )-(npred-ncorrect)) * 100. / batch_size, torch.zeros(1), torch.zeros(1), npred, ntrue ]
       
            
            if npred>0:
                apr[k][c][1] = ncorrect * 100. / npred
            if ntrue>0:
                apr[k][c][2] = ncorrect * 100. / ntrue




    return apr
