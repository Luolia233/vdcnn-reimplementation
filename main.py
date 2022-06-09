# -*- coding: utf-8 -*-
"""
@author: 
        - Luolia233  <723830981@qq.com>
@brief:
"""

import os
import argparse
from trainer import vdcnn_trainer

# 作用：将命令行解析成Python数据类型所需的全部信息
def get_args():
    # 设置解析器
    parser = argparse.ArgumentParser("""Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)""")
    # 添加参数
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--model_folder", type=str, default="ckpt/ag_news")
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news")
    parser.add_argument("--depth", type=int, choices=['9', '17', '29'], default='9', help="Depth of the network tested in the paper ('9', '17', '29')")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--solver", type=str, default="sgd", help="'agd' or 'adam'")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=15, help="Number of iterations before halving learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--nthreads", type=int, default=4)
    # 解析参数
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 获取参数，打印
    opt = get_args()
    print("parameters: {}".format(vars(opt)))

    # 创建路径用于保存模型权重信息、获取数据集。
    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)
    VDCNN_trainer = vdcnn_trainer(opt)
    VDCNN_trainer.build()