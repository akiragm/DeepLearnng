# coding: utf-8
import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np

##ファイルpath
dataset_dir = os.path.dirname(os.path.abspath(__file__))
vocab_file = 'ptb.vocab.pkl'
vocab_path = dataset_dir + '/' + vocab_file


#openファイル
if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
            
##pklの配列長確認
print(len(word_to_id))
print(len(id_to_word))
