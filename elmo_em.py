#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:55:42 2019

@author: aryank
"""

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import tensorflow_hub as hub
import tensorflow as tf
dfy = pd.read_csv('sentiment labelled sentences/yelp_labelled.txt',names=['sentence', 'label'], sep='\t')
dfa = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt',names=['sentence', 'label'], sep='\t')
dfi = pd.read_csv('sentiment labelled sentences/imdb_labelled.txt',names=['sentence', 'label'], sep='\t')
df = pd.concat([dfy,dfa,dfi])
#from sklearn.model_selection import train_test_split
#xtrain,xtest,ytrain,ytest = train_test_split(df['sentence'],df['label'])
#print(xtrain.shape)
#print(xtest.shape)
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
df['clean_sen'] = df['sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
df['clean_sen'] = df['clean_sen'].str.lower()
nlp = spacy.load("en_core_web_md")
# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output
df['clean_sen'] = lemmatization(df['clean_sen'])
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
x = ["Roasted ants are a popular snack in Columbia"]

# Extract ELMo features 
embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

embeddings.shape

