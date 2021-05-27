#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:28:33 2020

@author: shizi
"""

import json
file = '/Users/shizi/desktop/project/dialog/data/toy_data.json'
with open(file, 'r', encoding='utf8') as f:
    cont = json.load(f)
valid = cont['valid']
train = cont['train']
test = cont['test']

def predeal(corpus):
    for i in range(len(corpus)):
        temp = corpus[i]
        for j in range(len(temp)):
            temp[j] = temp[j].replace(' ','',)
    for i in range(len(corpus)):
        temp = '\n'.join(corpus[i])
        corpus[i] = temp
    return '\n\n'.join(corpus)

test2 = predeal(test)
valid2 = predeal(valid)
train2 = predeal(train)

with open('/Users/shizi/desktop/project/dialog/data/toy_test.txt','w',encoding='utf8') as f:
    f.write(test2)
with open('/Users/shizi/desktop/project/dialog/data/toy_train.txt','w',encoding='utf8') as f:
    f.write(train2)
with open('/Users/shizi/desktop/project/dialog/data/toy_valid.txt','w',encoding='utf8') as f:
    f.write(valid2)




