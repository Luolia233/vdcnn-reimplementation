# -*- coding: utf-8 -*-
"""
@author: 
        - Luolia233  <723830981@qq.com>
@brief:
"""

import numpy as np
import torch.nn as nn
import torch
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm


def get_metrics(cm, list_metrics):
    """Compute metrics from a confusion matrix (cm)
    cm: sklearn confusion matrix
    returns:
    dict: {metric_name: score}

    """
    dic_metrics = {}
    total = np.sum(cm)

    if 'accuracy' in list_metrics:
        out = np.sum(np.diag(cm))
        dic_metrics['accuracy'] = out/total

    if 'pres_0' in list_metrics:
        num = cm[0, 0]
        den = cm[:, 0].sum()
        dic_metrics['pres_0'] =  num/den if den > 0 else 0

    if 'pres_1' in list_metrics:
        num = cm[1, 1]
        den = cm[:, 1].sum()
        dic_metrics['pres_1'] = num/den if den > 0 else 0

    if 'recall_0' in list_metrics:
        num = cm[0, 0]
        den = cm[0, :].sum()
        dic_metrics['recall_0'] = num/den if den > 0 else 0

    if 'recall_1' in list_metrics:
        num = cm[1, 1]
        den = cm[1, :].sum()
        dic_metrics['recall_1'] =  num/den if den > 0 else 0

    return dic_metrics

def get_optimizer(solver,lr,momentum,net):
    assert solver in ['sgd', 'adam']
    if solver == 'sgd':
        print(" - optimizer: sgd")
        return torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum)
    elif solver == 'adam':
        print(" - optimizer: adam")
        return  torch.optim.Adam(net.parameters(), lr = lr)    
def get_scheduler(optimizer,lr_halve_interval,gamma):
    if lr_halve_interval and  lr_halve_interval > 0:
        print(" - lr scheduler: {}".format(lr_halve_interval))
        return torch.optim.lr_scheduler.StepLR(optimizer, lr_halve_interval, gamma=gamma, last_epoch=-1)
    else:
        return None

def predict(net,dataset,device,msg="prediction"):
    
    net.eval()

    y_probs, y_trues = [], []

    for iteration, (tx, ty) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):

        data = (tx, ty)
        data = [x.to(device) for x in data]
        out = net(data[0])
        ty_prob = F.softmax(out, 1) # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[1].detach().cpu().numpy())

    return np.concatenate(y_probs, 0), np.concatenate(y_trues, 0).reshape(-1, 1)


