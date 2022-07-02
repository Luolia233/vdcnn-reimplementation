#预处理中文数据集,需要根据路径加载csv文件

#导入数据处理的基础包
import torch
import numpy as np
import pandas as pd
import csv
import re
#导入用于计数的包
from collections import Counter
import os
import requests
import jieba
from tqdm import tqdm
from gensim.models import word2vec
from torch.utils.data import Dataset

#读入csv文件
def readCSVbyPandas(path):
    # 读取数据
    # 路径是各自数据路径的存放地址
    data = pd.read_csv(path, encoding="gbk")
    # 输出数据的一些相关信息
    #print("<<<<<data.info()>>>>>")
    #print(data.info())
    #print()
    # 看数据形状 (行数, 列数)
    #print("<<<<<data.shape>>>>>")    #(14209, 2)
    #print(data.shape)
    #print()
    #看data是什么类型的
    #print("<<<<<type(data)>>>>>")  #<class 'pandas.core.frame.DataFrame'>
    #print(type(data))
    #print()
    # 列标签 <Index>
    #print("<<<<<data.columns>>>>>")
    #print(data.columns)
    #print()
    # 观察数据格式，分别查看data的前五条数据和后五条数据
    #print(data.head())
    #print(data.tail())
    return data


#去除字母数字表情和其它字符
def clear_character(sentence):
    pattern1='[a-zA-Z0-9]'
    pattern2 = re.compile(u'[^\s1234567890:：' + '\u4e00-\u9fa5]+')
    pattern3='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    line1=re.sub(pattern1,'',sentence)   #去除英文字母和数字
    line2=re.sub(pattern2,'',line1)   #去除表情和其他字符
    line3=re.sub(pattern3,'',line2)   #去除去掉残留的冒号及其它符号
    new_sentence=''.join(line3.split()) #去除空白
    return new_sentence


# TODO: 使用结巴完成对每一个comment的分词
#     seg = jieba.lcut(content)
def comment_cut(content):
    seg = list(jieba.cut(content.strip()))
    return seg


class MyData(Dataset):
    def __init__(self, sentences, label):
        self.sentences = sentences
        self.label = label

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return torch.tensor(self.sentences[i]), torch.tensor(self.label[i])

#返回的训练集和测试集是MyData类型，sentences是200维*50词的词向量.
# label对应csv文件中情感倾向，csv中1表示积极，对应label为0；csv中2表示消极，对应label为1
# 每个词表示为200维向量，所以vdcnn中删除了nn.embedding的操作
def Processing_dataChin(path, data_folder, maxlen, nthreads):  #返回训练集、测试集、类别、向量维度
    # 如项目路径为D:/Code/WebPro，数据集路径为D:/Code/WebPro/train.csv和D:/Code/WebPro/test.csv，则path='D:/Code/WebPro'
    # 停用词表的位置如D:/Code/WebPro/stopWord.json，也是path+'/stopWord.json'，即path2
    # 训练好的word2vec模型的位置如D:/Code/WebPro/w2v04.model，即path3 = path + '/w2v04.model'

    print()
    #step1: 读取train.csv和test.csv。
    #csv的数据保存在data中，data为pandas的DataFrame格式
    tr_data = readCSVbyPandas(path+"/train.csv")
    print("成功读取训练集原始数据")
    te_data = readCSVbyPandas(path+"/test.csv")
    print("成功读取测试集原始数据")

    print()
    #step2: 去掉一些无用的字符，自行定一个字符几何，并从文本中去掉
    tr_data["comment_processed"] = tr_data['评论内容'].apply(clear_character)  # 增加了一列
    print("成功删除训练集原始数据的无用字符")
    te_data["comment_processed"] = te_data['评论内容'].apply(clear_character)  # 增加了一列
    print("成功删除测试集原始数据的无用字符")

    print()
    #step3:采用jieba分词
    tqdm.pandas(desc='apply')
    tr_data['comment_processed'] = tr_data['comment_processed'].progress_apply(comment_cut)
    print("成功对训练集进行分词")
    tqdm.pandas(desc='apply')
    te_data['comment_processed'] = te_data['comment_processed'].progress_apply(comment_cut)
    print("成功对测试集进行分词")

    print()
    #step4: 读取停用词表，并保存在列表中
    path2 = path + '/stopWord.json'
    with open(path2, "r", encoding='utf-8') as f:
        stopWords = f.read().split("\n")
    print("成功读取停用词表")
    #print("len(stopWords) = ", len(stopWords))

    print()

    # step5: 删除停用词
    # 去除停用词函数
    def rm_stop_word(wordList):   #, stopWords
        # your code, remove stop words
        # TODO
        # outstr = ''
        # 去停用词
        # for word in wordList:
        # if word not in stopWords:
        # if word != '\t':
        # outstr += word
        # outstr += " "
        # return outstr
        filtered_words = [word for word in wordList if word not in stopWords]
        return filtered_words
        # return " ".join(filtered_words)


    #print(tr_data.head())
    #tr_data['comment_processed'] = rm_stop_word(tr_data['comment_processed'], stopWords)
    tr_data['comment_processed'] = tr_data['comment_processed'].progress_apply(rm_stop_word)
    print("成功对训练集删除停用词")
    #print(tr_data.head())
    te_data['comment_processed'] = te_data['comment_processed'].progress_apply(rm_stop_word)
    #te_data['comment_processed'] = rm_stop_word(te_data['comment_processed'], stopWords)
    print("成功对测试集删除停用词")

    print()
    #step6: 用训练好的word2vec模型将所有句子转换为词向量
    path3 = path + '/w2v04.model'
    model = word2vec.Word2Vec.load(path3)
    #对训练集操作
    sentences = tr_data.iloc[:, 2]
    labels = tr_data.iloc[:, 0]
    tr_sequences = []  # 整个训练集或测试集的句子
    tr_labels = []
    # 将句子转化为200维*50词的向量
    for sentence in sentences:
        i = 0
        seq = []  # 200维*50词
        for w in sentence:
            if i == 50:
                break
            # print("w=", w, "的词向量值")
            # print(model.wv.get_vector(w))
            seq.append(model.wv.get_vector(w))
            i += 1
        while i < 50:
            seq.append([0] * 200)
            i += 1
        tr_sequences.append(seq)
    for label in labels:
        tr_labels.append(label-1)
    print("成功对训练集形成词向量")
    print("训练集评论数量为", len(tr_sequences), "  标签数量为", len(tr_labels))
    #print("tr_labels[:5]", tr_labels[:5])
    #print("type(tr_labels) = ", type(tr_labels))
    #对测试集操作
    sentences = te_data.iloc[:, 2]
    labels = te_data.iloc[:, 0]
    te_sequences = []  # 整个训练集或测试集的句子
    te_labels = []
    # 将句子转化为200维*50词的向量
    for sentence in sentences:
        i = 0
        seq = []  # 200维*50词
        for w in sentence:
            if i == 50:
                break
            # print("w=", w, "的词向量值")
            # print(model.wv.get_vector(w))
            seq.append(model.wv.get_vector(w))
            i += 1
        while i < 50:
            seq.append([0] * 200)
            i += 1
        te_sequences.append(seq)
    for label in labels:
        te_labels.append(label-1)
    print("成功对测试集形成词向量")
    #print(te_sequences[:5])
    print("测试集评论数量为", len(te_sequences), "  标签数量为", len(te_labels))
    print()

    return MyData(tr_sequences, tr_labels), MyData(te_sequences, te_labels), 2, 200

#Processing_dataChin("/", '', 50, 0)