# -*- coding: utf-8 -*-
"""
@author: 
        - Luolia233  <723830981@qq.com>
@brief:
"""

import numpy as np
import os
import tarfile
import shutil


# 英文句子转为小写
class CharLowercase():
    def transform(self, sentences):
        return [s.lower() for s in sentences]

# 英文句子向量化
class CharVectorizer():
    def __init__(self, maxlen=10, padding='pre', truncating='pre', alphabet="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/| #$%ˆ&*˜‘+=<>()[]{}"""):
        
        self.alphabet = alphabet    # 符号集合
        self.maxlen = maxlen        # maxlen用于规定统一句子的最大长度。
        self.padding = padding      # 定义补全方式
        self.truncating = truncating # 定义截断方式
        # 字符集每个字符映射到一个值。
        self.char_dict = {'_pad_': 0, '_unk_': 1, ' ': 2} # 字典中常用标识符的映射关系。_unk_: 低频词或未在词表中的词，_pad_: 补全字符。
        for i, k in enumerate(self.alphabet, start=len(self.char_dict)):
            self.char_dict[k] = i

    def transform(self,sentences):
        sequences = []
        for sentence in sentences:
            # 将长短不一的句子中每个字符根据定义好的映射，转换为映射后的值的集合。未找见默认为_unk_表示未知的符号。
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]
            # 句子中符号的数目，表示句子长度。
            length = len(seq)
            # 根据maxlen对句子进行截断，将所有句子统一为maxlen长度，pre表示取前段，post表示取后段。
            if self.truncating == 'pre':
                seq = seq[-self.maxlen:]
            elif self.truncating == 'post':
                seq = seq[:self.maxlen]
            # 如果当前句子不足maxlen，则进行补全。pre玩前补、post往后补。
            if length < self.maxlen:
                diff = np.abs(length - self.maxlen)
                if self.padding == 'pre':
                    seq = [self.char_dict['_pad_']] * diff + seq
                elif self.padding == 'post':
                    seq = seq + [self.char_dict['_pad_']] * diff
            sequences.append(seq)                

        return sequences        
    
    def get_params(self):
        params = vars(self)
        return params

# 检查数据存放目录是否已经按约定形式存放，如果数据集是以压缩包的形式存在则进行解包。
def checkdata(data_folder):
    if os.path.exists(data_folder):
        for f in ["test.csv", "train.csv"]: # 检查训练集和数据集文件
            if not os.path.exists(os.path.join(data_folder, f)):
                for file in os.listdir(data_folder): # 查找压缩包
                    if os.path.splitext(file)[-1] == ".gz":
                        untar(data_folder,file) # 解压
                        return True
                return False
        return True
    else:
        return False

# 解压缩操作
def untar(targetdir,f):
    print('untar..')
    # 拼接解压路径 目录+压缩包名称
    tardir = os.path.join(targetdir, os.path.basename(f).split(".")[0])
    # 打开解压目录
    tfile = tarfile.open(os.path.join(targetdir, f), 'r:gz')
    # 进行解压
    tfile.extractall(path=targetdir)
    tfile.close() # close
    # 将解压文件移动到目标目录
    for file in os.listdir(tardir):
        src = os.path.join(tardir, file)
        dst = os.path.join(targetdir, file)
        shutil.move(src, dst)
    shutil.rmtree(tardir) # 递归删除
    print("file has untared")
