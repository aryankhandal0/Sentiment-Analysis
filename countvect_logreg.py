#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:14:43 2019

@author: aryank
"""

import pandas as pd
import numpy as np
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
#Count Vectorizer
sentences = ['John likes ice cream', 'John hates chocolate.']
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=0, lowercase = False)
vect.fit(sentences)
print(vect.vocabulary_)
#train_test_split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(df['sentence'],df['label'],test_size=0.3)
vectorizer = CountVectorizer()
vectorizer.fit(xtrain)
Xtrain = vectorizer.transform(xtrain)
Xtest = vectorizer.transform(xtest)
#train_model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(Xtrain,ytrain)
preds = logreg.predict(Xtest)
#accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest,preds)
print("Accuracy for LOGREG: " + str(acc*100)+"%")

