# -*- coding: utf-8 -*-
"""
@author: Luolia233 <723830981@qq.com>
@brief:
"""

import sys
import csv
import lmdb
import torch
from utils.utils import *
from tqdm import tqdm
from torch.utils.data import Dataset

# for windows
csv.field_size_limit(min(sys.maxsize, 2147483646))

# 哈希表，存储数据集名称和类别数量的对应关系。
n_classes = {"ag_news": 4, "db_pedia": 14, "yelp_review": 5, "yelp_review_polarity": 2, "amazon_review_full": 5,
             "amazon_review_polarity": 2, "sogou_news": 5, "yahoo_answers": 10, "imdb": 2}


# 功能：加载csv中的数据
class TextDataset(object):
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_folder = "datasets/{}/raw".format(self.data_name)
        self.n_classes = n_classes[self.data_name]

        # 检查数据集
        if not checkdata(self.data_folder):
            raise Exception("please put {} raw dataset.tar.gz or [test.csv, train.csv] into {}".format(self.data_name,
                                                                                                       self.data_folder))
    # 读取csv文件
    def _generator(self, filename):
        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for line in reader:
                # 将title和description合并
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                yield sentence, label

    def load_train_data(self):
        return self._generator(os.path.join(self.data_folder, "train.csv"))

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "test.csv"))


class MyData(Dataset):

    def __init__(self, sentences, label):
        self.sentences = sentences
        self.label = label

    def __len__(self):
        return len(self.sentences)

    # 定义使用索引访问元素
    def __getitem__(self, i):
        return torch.tensor(self.sentences[i]), torch.tensor(self.label[i])


def Processing_Data(dataset, data_folder, maxlen, nthreads):
    # 根据数据集名称加载数据,得到TextDataset对象, 默认使用ag_news数据集。
    dataset = TextDataset(dataset)
    dataset_name = dataset.__class__.__name__ # 数据集名字
    n_classes = dataset.n_classes   # 类别数
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    # 准备英文数据的预处理
    charLowercase = CharLowercase()  # 功能是把所有字母转为小写。
    vectorizer = CharVectorizer(maxlen=maxlen, padding='post', truncating='post')  # 功能是将句子向量化
    n_tokens = len(vectorizer.char_dict) # tokens数

    # 得到训练集和数据集的样本数目
    tr_sentences = [txt for txt, lab in dataset.load_train_data()]
    te_sentences = [txt for txt, lab in dataset.load_test_data()]
    n_tr_samples = len(tr_sentences)
    n_te_samples = len(te_sentences)
    print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))

    # 加载训练集
    sentences_tr = []
    labels_tr = []
    for i, (sentence, label) in enumerate(
            tqdm(dataset.load_train_data(), desc="transform train...", total=n_tr_samples)):
        xtxt = vectorizer.transform(charLowercase.transform([sentence]))[0] # 向量化
        sentences_tr.append(xtxt)
        labels_tr.append(label)

    # 加载测试集
    sentences_te = []
    labels_te = []
    for i, (sentence, label) in enumerate(
            tqdm(dataset.load_test_data(), desc="transform test...", total=n_te_samples)):
        xtxt = vectorizer.transform(charLowercase.transform([sentence]))[0]
        sentences_te.append(xtxt)
        labels_te.append(label)

    return MyData(sentences_tr,labels_tr), MyData(sentences_te, labels_te), n_classes, n_tokens