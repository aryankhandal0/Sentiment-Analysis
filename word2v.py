#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:02:49 2019

@author: aryank
"""
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import nltk
#import and head, info
dfy = pd.read_csv('sentiment labelled sentences/yelp_labelled.txt',names=['sentence', 'label'], sep='\t')
dfa = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt',names=['sentence', 'label'], sep='\t')
dfi = pd.read_csv('sentiment labelled sentences/imdb_labelled.txt',names=['sentence', 'label'], sep='\t')
print(dfy.head())
print(dfa.head())
print(dfi.head())
df = pd.concat([dfy,dfa,dfi])
print(df.head())
print(df.info())
# word2vec
sent_f = []
word_tok=[]
from gensim.models import Word2Vec
import gensim
sentences = df['sentence']
word_lis =[]
for sentence in sentences:
    words = word_tokenize(sentence)
    word_lis.append(words)
model = Word2Vec(word_lis,size=50,window=10,min_count=2)
#words = list(model.wv.vocab)
#print(words)
model.save('word2vecmodel.bin')
print(model)

    