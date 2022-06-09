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



class Preprocessing():

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def transform(self, sentences):
        """
        sentences: list(str) 
        output: list(str)
        """
        return [s.lower() for s in sentences]
class CharVectorizer():
    def __init__(self, maxlen=10, padding='pre', truncating='pre', alphabet="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/| #$%ˆ&*˜‘+=<>()[]{}"""):
        
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating

        self.char_dict = {'_pad_': 0, '_unk_': 1, ' ': 2}
        for i, k in enumerate(self.alphabet, start=len(self.char_dict)):
            self.char_dict[k] = i

    def transform(self,sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        sequences = []

        for sentence in sentences:
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]
            
            if self.maxlen:
                length = len(seq)
                if self.truncating == 'pre':
                    seq = seq[-self.maxlen:]
                elif self.truncating == 'post':
                    seq = seq[:self.maxlen]

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


def list_to_bytes(l):
    return np.array(l).tobytes()


def list_from_bytes(string, dtype=np.int):
    return np.frombuffer(string, dtype=dtype)

def checkdata(data_folder):
    if os.path.exists(data_folder):
            for f in ["test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(data_folder, f)):
                    for file in os.listdir(data_folder):
                        if os.path.splitext(file)[-1] == ".gz":
                            print("tar.gz checked")
                            untar(data_folder,file)
                            return True
                    return False
            return True
    else:
        return False
def untar(targetdir,f):
    print('Untaring file...')
    tardir = os.path.join(targetdir, os.path.basename(f).split(".")[0])
    tfile = tarfile.open(os.path.join(targetdir, f), 'r:gz')
    tfile.extractall(path=targetdir)
    tfile.close()
    for file in os.listdir(tardir):
        src = os.path.join(tardir, file)
        dst = os.path.join(targetdir, file)
        shutil.move(src, dst)
    shutil.rmtree(tardir)
    print("file untared")
