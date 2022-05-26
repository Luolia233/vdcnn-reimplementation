# -*- coding: utf-8 -*-
"""
@author: Luolia233 <723830981@qq.com>
@brief:
"""

import os
import sys
import csv
import tarfile
import shutil
import lmdb
from utils.utils import *
from tqdm import tqdm
from torch.utils.data import Dataset
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

def untar(untar_fpath,datadir,fpath):
    if not os.path.exists(untar_fpath):
        print('Untaring file...')
        tfile = tarfile.open(fpath, 'r:gz')
        try:
            tfile.extractall(path=datadir)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(untar_fpath):
                if os.path.isfile(untar_fpath):
                    os.remove(untar_fpath)
                else:
                    shutil.rmtree(untar_fpath)
            raise
        tfile.close()
    return untar_fpath


class TupleLoader(Dataset):

    def __init__(self, path=""):
        self.path = path

        self.env = lmdb.open(path, max_readers=opt.nthreads, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        return xtxt, lab



def LoadData(dataset,data_folder,maxlen):
    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    tr_path =  "{}/train.lmdb".format(opt.data_folder)
    te_path = "{}/test.lmdb".format(opt.data_folder)
    
    # check if datasets exis
    all_exist = True if (os.path.exists(tr_path) and os.path.exists(te_path)) else False

    preprocessor = Preprocessing()
    vectorizer = CharVectorizer(maxlen=opt.maxlen, padding='post', truncating='post')
    n_tokens = len(vectorizer.char_dict)

    if not all_exist:
        print("Creating datasets")
        tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_sentences = [txt for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
            
        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)
        del tr_sentences
        del te_sentences

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))

        ###################
        # transform train #
        ###################
        with lmdb.open(tr_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):

                    xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i
                    
                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        ##################
        # transform test #
        ##################
        with lmdb.open(te_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):

                    xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i
                    
                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))
    return TupleLoader(tr_path),TupleLoader(te_path),n_classes,n_tokens
