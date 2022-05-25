# -*- coding: utf-8 -*-
"""
@author: Luolia233 <723830981@qq.com>
@brief:
"""

import os
import sys
import csv

from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
n_classes = {"ag_news":4,"db_pedia":14,"yelp_review":5,"yelp_review_polarity":2,"amazon_review_full":5,"amazon_review_polarity":2,"sogou_news":5,"yahoo_answers":10,"imdb":2}


class TextDataset(object):

    def __init__(self,data_name):
        self.data_name = data_name
        self.data_folder = "datasets/raw/{}".format(self.data_name)
        self.n_classes = n_classes[self.data_name]        

        # 检查数据集
        if os.path.exists(self.data_folder):
            for f in ["classes.txt", "readme.txt", "test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(self.data_folder, f)):
                    print("please put {} raw dataset into {}".format(self.data_name,self.data_folder))
        else:
            print("please put {} raw dataset into {}".format(self.data_name,self.data_folder))

    def _generator(self, filename):
        if self.data_name == "imdb":
            with open(filename, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f, quotechar='"')
                for line in reader:
                    sentence = line['sentence']
                    label = int(line['label'])
                    # if sentence and label:
                    yield sentence, label
        else:
            with open(filename, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
                for line in reader:
                    sentence = "{} {}".format(line['title'], line['description'])
                    label = int(line['label']) - 1
                    yield sentence, label

    def load_train_data(self):
        return self._generator(os.path.join(self.data_folder, "train.csv"))

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "test.csv"))


def load_datasets(names=["ag_news", "imdb"]):
    """
    Select datasets based on their names

    :param names: list of string of dataset names
    :return: list of dataset object
    """

    datasets = []

    if 'ag_news' in names:
        datasets.append(TextDataset("ag_news"))
    if 'db_pedia' in names:
        datasets.append(TextDataset("db_pedia"))
    if 'yelp_review' in names:
        datasets.append(TextDataset("yelp_review"))
    if 'yelp_polarity' in names:
        datasets.append(TextDataset("yelp_polarity"))
    if 'amazon_review' in names:
        datasets.append(TextDataset("amazon_review"))
    if 'amazon_polarity' in names:
        datasets.append(TextDataset("amazon_polarity"))
    if 'sogou_news' in names:
        datasets.append(TextDataset("sogou_news"))
    if 'yahoo_answer' in names:
        datasets.append(TextDataset("yahoo_answer"))
    if 'imdb' in names:
        datasets.append(TextDataset("imdb"))
    return datasets


if __name__ == "__main__":

    names = [
        'imdb',
        'ag_news',
        'db_pedia',
        'yelp_review',
        'yelp_polarity',
        'amazon_review',
        'amazon_polarity',
        'sogou_news',
        'yahoo_answer',
    ]

    for name in names:
        print("name: {}".format(name))
        dataset = load_datasets(names=[name])[0]
        
        # train data generator
        gen = dataset.load_train_data()
        sentences, labels = [], []
        for sentence, label in tqdm(gen):
            sentences.append(sentence)
            labels.append(label)
        print(" train: (sentences,labels) = ({}/{})".format(len(sentences), len(labels)))

        # test data generator
        gen = dataset.load_test_data()
        sentences, labels = [], []
        for sentence, label in tqdm(gen):
            sentences.append(sentence)
            labels.append(label)
        print(" train: (sentences,labels) = ({}/{})".format(len(sentences), len(labels)))
