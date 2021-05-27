#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:24:18 2020

@author: shizi
"""

from tqdm import tqdm
import pdb
import torch

def split(length, sublength ):
    '''函数用于分割length,每份的长度为sublength，不足的放到新的一份中
    exam:
         split(125,20) = [(0,20),(20,40),(40,60),(60,80),(80,100),(100,120),(120,125)]'''
    sur = list(range(0,length,sublength))
    sur += [length]
    sur2 = []
    for i in range(len(sur)-1):
        sur2.append((sur[i],sur[i+1]))
    return sur2

def Lccd2Tensor(corpus, flag, totbit, tokenizer, maxlong=None, ):
    '''函数将原始语料corpus转化成类似数组的list格式'''
    
    pbar = tqdm(corpus)
    ChCorpus = []
    pbar.set_description("We are tranfer the batch data, %d/%d"%(flag+1,totbit+1))
    for i in pbar:
        append = []
        for j in i:
            append2 = list(j.replace(' ',''))
            append.append(append2)
        ChCorpus.append(append)
    pbar.close()

    # 闲聊对话句子拼接    
    ChCorpus2 = []
    pbar = tqdm(ChCorpus)
    pbar.set_description("正在拼接句子")
    for i in pbar:
        temp = []
        temp.append("[CLS]")
        for j in i:
            temp.extend(j)
            temp.append("[SEP]")

        ChCorpus2.append(temp)
    pbar.close()

    
    # 将文本转id
    ChCorpus3 = []
    pbar = tqdm(range(len(ChCorpus2)))
    pbar.set_description("正在转换到张量模式")
    for i in pbar:
        temp = ChCorpus2[i]
        content = []
        for j in range(len(temp)):
            if j >= maxlong:
                break
            else:
                try:
                    inde = tokenizer.convert_tokens_to_ids(temp[j])
                    content.append(inde)
                except:
                    inde = tokenizer.unk_token
                    content.append(inde)
        ChCorpus3.append(content)            
    pbar.close()
    return ChCorpus3

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = int(end_time - start_time)
    elapsed_hour = elapsed_time // 3600
    elapsed_mins = (elapsed_time % 3600) // 60
    elapsed_secs = (elapsed_time % 3600) % 60
    return elapsed_hour, elapsed_mins, elapsed_secs

def savemodel(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model,f)


def mycollate_fn(batch):
    '''计算该batch最大长度，并将batch内所有样本用补全0的方式统一长度
       batch format:  sample
                        ⬇️      
          content -> [2, 16454, 3, 11395, 37411, 80222, 17822, 76710, 3, 46016, 108468, 3]
                     [2, 39125, 37854, 1550, 3, 41783, 13210, 2567, 24591, 3]
    '''
    maxlong = 0
    input_batch = []
    for i in batch:
        if len(i) > maxlong:
            maxlong = len(i)
    for i in range(len(batch)):
        input_len = len(batch[i])
        input_batch.append(batch[i])
        input_batch[i].extend([0] * (maxlong - input_len))
        if sum([t==3 for t in input_batch[i]]) == 0:
            input_batch[i][input_len//2] = 3
            
    return torch.tensor(input_batch, dtype = torch.long).T