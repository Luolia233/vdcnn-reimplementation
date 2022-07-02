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
        self.downsample = (input_dim != output_dim)
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


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class Res2NetBottleneck(nn.Module):
    expansion = 1 
    def __init__(self, inplanes, planes, stride=1, scales=4, groups=1, se=False,  norm_layer=True):
        #scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Res2NetBottleneck, self).__init__()

        if planes % scales != 0: #输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  #BN层
            norm_layer = nn.BatchNorm1d

        self.downsample = (inplanes != planes)
        bottleneck_planes = groups * planes
        self.scales = scales
        self.stride = stride
        self.conv1 = nn.Conv1d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([nn.Conv1d(bottleneck_planes // scales, bottleneck_planes // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = nn.Conv1d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.ds = nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(planes))

        #SE模块
        self.se = SEModule(planes * self.expansion) if se else None

    def forward(self, x):
        identity = x

        #1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, 1) #将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        #1*1的卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        #加入SE模块
        if self.se is not None:
            out = self.se(out)
        #下采样
        if self.downsample:
            identity = self.ds(identity)

        out += identity
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

# class VDCNN(nn.Module):
#
#     def __init__(self, n_classes=2, table_in=141, table_out=16, depth='9', shortcut=False, convblock='res2net_style'):
#         super(VDCNN, self).__init__()
#
#         self.convblock = convblock
#         self.embed = nn.Embedding(table_in, table_out, padding_idx=0, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
#
#         self.layers = self._make_layers(cfg[depth],table_out,shortcut)
#
#         fc_layers = []
#         fc_layers += [nn.Linear(4096, 2048), nn.ReLU()]
#         fc_layers += [nn.Linear(2048, 2048), nn.ReLU()]
#         fc_layers += [nn.Linear(2048, n_classes)]
#
#
#         self.fc_layers = nn.Sequential(*fc_layers)
#
#         self.__init_weights()
#
#     def _make_layers(self, cfg,table_out,shortcut):
#         layers = []
#         layers += [nn.Conv1d(table_out, 64, kernel_size=3, padding=1)]
#         in_channels = 64
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]
#             else:
#                 if self.convblock == 'resnet_style':
#                     layers += [BasicConvBlock(input_dim=in_channels, output_dim=x, kernel_size=3, padding=1, shortcut=shortcut)]
#                 else:
#                     layers += [Res2NetBottleneck(inplanes=in_channels, planes=x)]
#                 in_channels = x
#         layers += [nn.AdaptiveMaxPool1d(8)]
#         return nn.Sequential(*layers)
#
#
#     def __init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#
#         out = self.embed(x)
#         out = out.transpose(1, 2)
#         out = self.layers(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc_layers(out)
#
#         return out

class VDCNN(nn.Module):

    def __init__(self, n_classes=2, table_in=141, table_out=16, depth='9', shortcut=False, convblock='res2net_style'):
        super(VDCNN, self).__init__()

        self.convblock = convblock

        if(table_in == 200):  #中文词向量已形成，每个词200维，因此无需nn.Embedding形成词向量
            self.embed = None
        else:    #英文数据集的情况
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
                if self.convblock == 'resnet_style':
                    layers += [BasicConvBlock(input_dim=in_channels, output_dim=x, kernel_size=3, padding=1, shortcut=shortcut)]
                else:
                    layers += [Res2NetBottleneck(inplanes=in_channels, planes=x)]
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
        if(self.embed):
            out = self.embed(x)
        else:
            out = x
        out = out.transpose(1, 2)
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)

        return out