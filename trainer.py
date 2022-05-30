# -*- coding: utf-8 -*-
"""
@author: 
        - Luolia233  <723830981@qq.com>
@brief:
"""

from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import LoadData
from network.vdcnn import VDCNN

from utils.nn_tools import *





def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,scheduler=None,criterion=None,list_metrics=[]):
    
    net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses,nclasses), dtype=int)

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx, ty) in enumerate(dataset):
            
            data = (tx, ty)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0])
            ty_prob = F.softmax(out, 1) # probabilites

            #metrics
            y_true = data[1].detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1].cpu().numpy()

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)
            
            loss =  criterion(out, data[1]) 
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(iteration+1)

            if optimize:
                loss.backward()
                optimizer.step()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()

def run(opt):
    traindata,testdata,n_classes,n_tokens = LoadData(opt.dataset,opt.data_folder,opt.maxlen,opt.nthreads)

    tr_loader = DataLoader(traindata, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nthreads, pin_memory=True)
    te_loader = DataLoader(testdata, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, pin_memory=False)

    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy']


    print("Creating model...")
    net = VDCNN(n_classes=n_classes, num_embedding=n_tokens + 1, embedding_dim=16, depth=opt.depth, n_fc_neurons=2048, shortcut=opt.shortcut)
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)

    assert opt.solver in ['sgd', 'adam']
    if opt.solver == 'sgd':
        print(" - optimizer: sgd")
        optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr, momentum=opt.momentum)
    elif opt.solver == 'adam':
        print(" - optimizer: adam")
        optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr)    
        
    scheduler = None
    if opt.lr_halve_interval and  opt.lr_halve_interval > 0:
        print(" - lr scheduler: {}".format(opt.lr_halve_interval))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)
        
    for epoch in range(1, opt.epochs + 1):
        train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion,list_metrics=list_metrics)
        train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
            print("snapshot of model saved as {}".format(path))
            save(net, path=path)


    if opt.epochs > 0:
        path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
        print("snapshot of model saved as {}".format(path))
        save(net, path=path)
