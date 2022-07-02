# -*- coding: utf-8 -*-
"""
@author: 
        - Luolia233  <723830981@qq.com>
@brief:
"""

import os
import argparse
from trainer import vdcnn_trainer
from datasets import Processing_Data
from network.vdcnn import VDCNN
from datasetChin import Processing_dataChin

# 作用：将命令行解析成Python数据类型所需的全部信息
def get_args():
    # 设置解析器
    parser = argparse.ArgumentParser("""Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)""")
    # 添加参数
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--model_folder", type=str, default="ckpt/ag_news")
    parser.add_argument("--depth", type=str, choices=['9', '17', '29'], default='9', help="Depth of the network tested in the paper ('9', '17', '29')")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--maxlen_c", type=int, default=50,help='chinese sentences needs pre 50 words')
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--solver", type=str, default="sgd", help="'agd' or 'adam'")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=15, help="Number of iterations before halving learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gpuid", type=int, default=-1)
    parser.add_argument("--list_metrics", type=str, nargs='+', default=["accuracy"], help="Compute metrics from a confusion matrix")
    parser.add_argument("--lmdb_nthreads", type=int, default=1, help="If the program runs on your own PC, maybe it needs to be set 1")
    parser.add_argument("--num_workers", type=int, default=0, help="If the program runs on your own PC, maybe it needs to be set 0")
    parser.add_argument("--language_dataset",type=str,default='chinese',help="According the language you choose to process data")
    # 解析参数
    args = parser.parse_args()
    return args

def main():
    # 获取参数，打印
    opt = get_args()
    print("parameters: {}".format(vars(opt)))

    # 创建路径用于保存模型权重信息、获取数据集。    
    data_folder = "datasets/{}/raw".format(opt.dataset)
    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    if opt.language_dataset == 'chinese':
        # 数据预处理
        print("Processing data...")
        trainset, testset, n_classes, n_tokens = Processing_dataChin("D:/vdcnn-reimplementation", '', opt.maxlen,
                                                                     opt.lmdb_nthreads)
        # 创建模型
        print("Creating model...")
        net = VDCNN(n_classes=n_classes, table_in=200, table_out=200, depth=opt.depth, shortcut=opt.shortcut)
        # 训练配置
        VDCNN_trainer = vdcnn_trainer(opt, net, trainset, testset, n_classes)
        # 模型训练
        VDCNN_trainer.build()
    else:
        # 数据预处理
        print("Processing data...")
        trainset, testset, n_classes, n_tokens = Processing_Data(opt.dataset, data_folder, opt.maxlen,
                                                                 opt.lmdb_nthreads)
        # 创建模型
        print("Creating model...")
        net = VDCNN(n_classes=n_classes, table_in=n_tokens + 1, table_out=16, depth=opt.depth, shortcut=opt.shortcut)
        # 训练配置
        VDCNN_trainer = vdcnn_trainer(opt,net,trainset,testset,n_classes)
        # 模型训练
        VDCNN_trainer.build()

if __name__ == "__main__":
    main()