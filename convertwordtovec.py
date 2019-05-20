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
#def preprocess_x(data):
#    cdf = pd.DataFrame(data)
#    cdf.reset_index(inplace=True)
#    cdf.drop('index',axis=1,inplace=True)
#    return cdf
def convtraintest2vec(data):
    i = 0
    ind = 0
    sent_list_vec = []
    list_ind = []
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
    return sent_list_vec, list_ind
#def preprocess_y(data,list_ind):
#    cdf = pd.DataFrame(data)
#    cdf.reset_index(inplace=True)
#    cdf.drop('index',axis=1,inplace=True)
#    cdf.drop(list_ind,inplace=True)
#    return list(cdf)
#xtrain = preprocess_x(xtrain)
#xtest = preprocess_x(xtest)
sent_train, train_missing = convtraintest2vec(xtrain)
sent_test, test_missing = convtraintest2vec(xtest)
#trainY = preprocess_y(ytrain,train_missing)
#testY = preprocess_y(ytest,test_missing)
maxi = []
for l in sent_train:
    maxi.append(len(l))
max_l = max(maxi)

trainX = pad_sequences(sent_train,maxlen = max_l)
testX = pad_sequences(sent_test,maxlen = max_l)
print("Xtrain_Original: " + str(len(xtrain)))
print("Xtest_Original: " + str(len(xtest)))
print("Xtrain_Preprocessed: " + str(len(trainX)))
print("Xtest_Preprocessed: " + str(len(testX)))
print("Index missing in train: " + str(list(train_missing)))
print("Index missing in test: " + str(list(test_missing)))


    

        
        
        
    