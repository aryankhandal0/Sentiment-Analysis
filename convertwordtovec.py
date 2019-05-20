#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:34 2019

@author: aryank
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
#import and head, info
dfy = pd.read_csv('sentiment labelled sentences/yelp_labelled.txt',names=['sentence', 'label'], sep='\t')
dfa = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt',names=['sentence', 'label'], sep='\t')
dfi = pd.read_csv('sentiment labelled sentences/imdb_labelled.txt',names=['sentence', 'label'], sep='\t')
df = pd.concat([dfy,dfa,dfi])
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(df['sentence'],df['label'])
w2v = Word2Vec.load('word2vecmodel.bin')
def convtraintest2vec(data):
    i = 0
    ind = 0
    sent_list_vec = []
    for x in xtrain:
        ind = ind + 1
        sent_list = []
        words = word_tokenize(x)
        for word in words:
            flag=0
            try:
                vector = w2v[word]
                flag=1
                for vec in vector:
                    sent_list.append(vec)
            except Exception as e:
                i = i+1
                print(e)
        if flag:
            sent_list_vec.append(sent_list)
        else:
            list_ind.append(ind)
    return 
    sent_list_vec, list_ind
sent_train, train_missing = convtraintest2vec(xtrain)
sent_test, test_missing = convtraintest2vec(xtest)
#i = 0
#ind = 0
#sent_list_vec = []
#for x in xtrain:
#    ind = ind + 1
#    sent_list = []
#    words = word_tokenize(x)
#    for word in words:
#        flag=0
#        try:
#            vector = w2v[word]
#            flag=1
#            for vec in vector:
#                sent_list.append(vec)
#        except Exception as e:
#            i = i+1
#            #print(e)
#    if flag:
#        sent_list_vec.append(sent_list)
#i = 0
#ind = 0
#sent_list_vec_test = []
#for x in xtest:
#    ind = ind + 1
#    sent_list = []
#    words = word_tokenize(x)
#    for word in words:
#        flag=0
#        try:
#            vector = w2v[word]
#            flag=1
#            for vec in vector:
#                sent_list.append(vec)
#        except Exception as e:
#            i = i+1
#            print(e)
#    if flag:
#        sent_list_vec_test.append(sent_list)
#    else:
#        list_ind.append(ind)
        
print(np.array(sent_list_vec).shape)
print(np.array(sent_list_vec_test).shape)
print(ytrain.shape)
print(ytest.shape)
print(len(xtrain))
print(len(ytrain))
maxi = []
for l in sent_list_vec:
    maxi.append(len(l))
max_l = max(maxi)
trainX = pad_sequences(sent_list_vec,maxlen = max_l)
print(trainX.shape)
testX = pad_sequences(sent_list_vec_test,maxlen = max_l)
print(testX.shape)
print(list_ind)


    

        
        
        
    