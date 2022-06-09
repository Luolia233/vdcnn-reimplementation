# -*- coding: utf-8 -*-
"""
@author: 
        - Luolia233  <723830981@qq.com>
@brief:
"""

from torch.utils.data import DataLoader

from datasets import LoadData
from network.vdcnn import VDCNN

from utils.nn_tools import *


class vdcnn_trainer():
    def __init__(self, opt):
        self.model_folder = opt.model_folder
        self.epochs = opt.epochs
        self.snapshot_interval = opt.snapshot_interval
        self.device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")


        print("Loading data...")
        traindata,testdata,n_classes,n_tokens = LoadData(opt.dataset,opt.data_folder,opt.maxlen,opt.nthreads)
        self.tr_loader = DataLoader(traindata, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nthreads, pin_memory=True)
        self.te_loader = DataLoader(testdata, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, pin_memory=False)
        self.n_classes = n_classes


        print("Creating model...")
        self.net = VDCNN(n_classes=n_classes, table_in=n_tokens + 1, table_out=16, depth=opt.depth, shortcut=opt.shortcut)
        self.net.to(self.device)

        self.optimizer = get_optimizer(opt.solver,opt.lr,opt.momentum,self.net)
        self.scheduler = get_scheduler(self.optimizer,opt.lr_halve_interval,opt.gamma)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.list_metrics = ['accuracy']

    def train(self,epoch):
        
        self.net.train()
        epoch_loss = 0
        cm = np.zeros((self.n_classes,self.n_classes), dtype=int)

        with tqdm(total=len(self.tr_loader),desc="Epoch {} - {}".format(epoch, "training")) as pbar:
            for iteration, (tx, ty) in enumerate(self.tr_loader):
                
                data = (tx, ty)
                data = [x.to(self.device) for x in data]

                self.optimizer.zero_grad()

                out = self.net(data[0])
                ty_prob = F.softmax(out, 1) # probabilites

                #metrics
                y_true = data[1].detach().cpu().numpy()
                y_pred = ty_prob.max(1)[1].cpu().numpy()

                cm += metrics.confusion_matrix(y_true, y_pred, labels=range(self.n_classes))
                dic_metrics = get_metrics(cm, self.list_metrics)
                
                loss =  self.criterion(out, data[1]) 
                epoch_loss += loss.item()
                dic_metrics['logloss'] = epoch_loss/(iteration+1)


                loss.backward()
                self.optimizer.step()
                dic_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']

                pbar.update(1)
                pbar.set_postfix(dic_metrics)

        self.scheduler.step()

    def test(self,epoch):
        
        self.net.eval()
        epoch_loss = 0
        cm = np.zeros((self.n_classes,self.n_classes), dtype=int)

        with tqdm(total=len(self.te_loader),desc="Epoch {} - {}".format(epoch, "testing")) as pbar:
            for iteration, (tx, ty) in enumerate(self.te_loader):
                
                data = (tx, ty)
                data = [x.to(self.device) for x in data]


                out = self.net(data[0])
                ty_prob = F.softmax(out, 1) # probabilites

                #metrics
                y_true = data[1].detach().cpu().numpy()
                y_pred = ty_prob.max(1)[1].cpu().numpy()

                cm += metrics.confusion_matrix(y_true, y_pred, labels=range(self.n_classes))
                dic_metrics = get_metrics(cm, [])
                
                loss =  self.criterion(out, data[1]) 
                epoch_loss += loss.item()
                dic_metrics['logloss'] = epoch_loss/(iteration+1)


                pbar.update(1)
                pbar.set_postfix(dic_metrics)


    def build(self):

        for epoch in range(self.epochs):
            self.train(epoch)
            self.test(epoch)

            if ((epoch+1) % self.snapshot_interval == 0) and (epoch > 0):
                path = "{}/model_epoch_{}".format(self.model_folder,epoch)
                print("snapshot of model saved as {}".format(path))
                torch.save(self.net.state_dict(),path+'.pt')

        path = "{}/model_epoch_{}".format(self.model_folder,self.epochs)
        print("final model saved as {}".format(path))
        torch.save(self.net.state_dict(),path+'.pt')
