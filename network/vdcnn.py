# -*- coding: utf-8 -*-
"""
@author: Luolia233 <723830981@qq.com>
@brief:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Convolutional Block
VDCNN的基本结构,每个卷积块由两个卷积层,经过BN和relu后构成
X → conv → BN → relu
                  ↓
                 conv → BN → relu → 



论文中的可选设置:resnet的shortcut
 X ---------------------→ downsample 
 ↓                            ↓
conv → BN → relu              ↓
                  ↓           ↓
                 conv → BN → relu → 

"""
class BasicConvBlock(nn.Module):

    def __init__(self, input_dim=128, output_dim=256, kernel_size=3, padding=1, stride=1, shortcut=False):
        super(BasicConvBlock, self).__init__()
        self.downsample = (input_dim == output_dim)
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.ds = nn.Sequential(nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(output_dim))

    def forward(self, x):

        residual = x
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            if self.downsample:
                residual = self.ds(x)
            out += residual

        out = self.relu(out)

        return out


"""
VDCNN的总体结构,复现了论文中的3种深度:[9,17,29]
                    X
                    ↓
                embedding
                    ↓
                  conv
                    ↓
                ConvBlocks1
                    ↓
                 MaxPool
                    ↓
                ConvBlocks2
                    ↓
                 MaxPool
                    ↓
                ConvBlocks3
                    ↓
                 MaxPool
                    ↓
                ConvBlocks4
                    ↓
              k-max pooling
                    ↓
              fc(4096,2048)
                    ↓
              fc(4096,2048)
                    ↓
            fc(4096,n_classes)
"""
cfg = {
    '9': [64, 'M', 128, 'M', 256, 'M',  512],
    '17': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
    '29': [64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 'M', 256, 256, 'M', 512, 512],
}

class VDCNN(nn.Module):

    def __init__(self, n_classes=2, table_in=141, table_out=16, depth='9', shortcut=False):
        super(VDCNN, self).__init__()

        self.embed = nn.Embedding(table_in, table_out, padding_idx=0, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)

        self.layers = self._make_layers(cfg[depth],table_out,shortcut)

        fc_layers = []
        fc_layers += [nn.Linear(4096, 2048), nn.ReLU()]
        fc_layers += [nn.Linear(2048, 2048), nn.ReLU()]
        fc_layers += [nn.Linear(2048, n_classes)]


        self.fc_layers = nn.Sequential(*fc_layers)

        self.__init_weights()

    def _make_layers(self, cfg,table_out,shortcut):
        layers = []
        layers += [nn.Conv1d(table_out, 64, kernel_size=3, padding=1)]
        in_channels = 64
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]
            else:
                layers += [BasicConvBlock(input_dim=in_channels, output_dim=x, kernel_size=3, padding=1, shortcut=shortcut)]
                in_channels = x
        layers += [nn.AdaptiveMaxPool1d(8)]
        return nn.Sequential(*layers)


    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        out = self.embed(x)
        out = out.transpose(1, 2)
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)

        return out

