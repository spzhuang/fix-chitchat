#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:37:42 2020

@author: shizi
"""


import torch
import os
import json
from tqdm import tqdm
import numpy as np
import pdb
import math
from torch.utils.data.dataloader import DataLoader
import datetime
from util_data import split, Lccd2Tensor, mycollate_fn
Tensor = torch.Tensor
F = torch.functional.F
import transformers

model_file = "/Users/shizi/desktop/project/GPT2-chit/trained-models/dialogue_model/"
corpus_file = "/Users/shizi/desktop/dataset/LCCD-DEAL/LCCD_1~10.json"
file_w2i = "/Users/shizi/desktop/dataset/LCCD-DEAL/word2idx_11.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = transformers.GPT2LMHeadModel.from_pretrained(model_file).to(device)

with open(corpus_file,'r',encoding='utf8') as f:
    corpus = json.load(f)
with open(file_w2i,'r',encoding = 'utf8') as f:
    word2idx = eval(f.read())
if "EOS" in word2idx:
    word2idx.pop("EOS")
idx2word = {v:k for k,v in word2idx.items()}
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')

def tensor2string(tensor:list,idx2word=idx2word):
    '''tensor 应该是单列表，而非嵌套列表'''
    string = ''
    tensor = tensor.tolist()
    for i in tensor:
        if i != 0:
            string += tokenizer.convert_ids_to_tokens(i)
    return string



def rouge_l(predict:Tensor, target:Tensor):
    '''
    --------
    使用时要检查一下，针对的是哪一个tokenizer
    '''
    all_sep = sum(target == 102).item()
    target = target.tolist()
    target = [i for i in target if i != 0]
    arange_for_predict = torch.arange(len(predict))
    all_sep_index = arange_for_predict[predict == 102]
    if len(all_sep_index) >= all_sep:
        index = all_sep_index[all_sep-1].item()
    elif len(all_sep_index)>0:
        index = all_sep_index[-1].item()
    else:
        index = len(predict)-1
    predict = predict.tolist()
    predict = predict[:index+1]
    
    length_predict, length_target = len(predict), len(target)
    
    target = [0]+target
    predict = [0]+predict
    init_array = np.zeros([len(predict),len(target)],dtype = int)
    
    
    for i in range(init_array.shape[0]):
        for j in range(init_array.shape[1]):
            if i>0 and j>0:
                if predict[i] == target[j]:
                    init_array[i,j] = init_array[i-1,j-1]+1
                else:
                    init_array[i,j] = max(init_array[i-1,j],init_array[i,j-1])
    lcs = init_array[-1,-1]
    if lcs == 0:
        r_lcs, p_lcs, f_lcs = 0, 0, 0
    else:
        r_lcs, p_lcs = lcs/length_target, lcs/length_predict
        f_lcs = (2*r_lcs*p_lcs)/(r_lcs+p_lcs)
    return r_lcs, p_lcs, f_lcs

def batch_rouge_l(pre_batch, rea_batch):

    r_lcs,p_lcs,f_lcs = 0, 0, 0
    assert len(pre_batch) == len(rea_batch)
    all_sample = len(pre_batch)
    for i in range(len(pre_batch)):
        temp_r, temp_p, temp_f = rouge_l(pre_batch[i],rea_batch[i])
        r_lcs += temp_r
        p_lcs += temp_p
        f_lcs += temp_f
        if np.isnan(temp_f):
            pdb.set_trace()
    if np.isnan(f_lcs):
        pdb.set_trace()
    return r_lcs/all_sample, p_lcs/all_sample, f_lcs/all_sample



def s_bleu(predict:Tensor, target:Tensor, N, W = None):
    
    assert N == 4 or N == 2, "N must be 2 or 4"
    all_sep = sum(target == 102).item()
    target = target.tolist()
    target = [i for i in target if i != 0]
    arange_for_predict = torch.arange(len(predict))
    all_sep_index = arange_for_predict[predict == 102]
    if len(all_sep_index) >= all_sep:
        index = all_sep_index[all_sep-1].item()
    elif len(all_sep_index)>0:
        index = all_sep_index[-1].item()
    else:
        index = len(predict)-1
    predict = predict.tolist()
    predict = predict[:index+1]
    
    if not W:
        W = 1/N
    cp = [0.0001 for i in range(N)]
    total_cp = [len(predict)-i for i in range(N)]
    # 针对N=1
    for i in predict:
        if i in target:
            cp[0] = int(cp[0])+1 

    # 针对N=2
    if total_cp[1] <= 0:
        total_cp[1] = 0
    else:
        predict2 = [(predict[i],predict[i+1]) for i in range(len(predict) - 1)]
        target2 = [(target[i],target[i+1]) for i in range(len(target) - 1)]
        for i in predict2:
            if i in target2:
                cp[1] = int(cp[1])+1
    
    if N == 4:
        # 针对N=3
        if total_cp[2] <= 0:
            total_cp[2] = 0
        else:
            predict3 = [(predict[i],predict[i+1],predict[i+2]) for i in range(len(predict) - 2)]
            target3 = [(target[i],target[i+1],target[i+2]) for i in range(len(target) - 2)]
            for i in predict3:
                if i in target3:
                    cp[2] = int(cp[2])+1
        
        # 针对N=4
        if total_cp[3] <= 0:
            total_cp[3] = 0
        else:
            predict4 = [(predict[i],predict[i+1],predict[i+2],predict[i+3]) for i in range(len(predict) - 3)]
            target4 = [(target[i],target[i+1],target[i+2],target[i+3]) for i in range(len(target) - 3)]
            for i in predict4:
                if i in target4:
                    cp[3] = int(cp[3])+1
    elif N == 2:
        pass
    app = [cp[i]/total_cp[i] if total_cp[i]>0 else 0 for i in range(N)]
 
    BP = 1 if len(predict)>len(target) else math.exp(1-len(target)/len(predict))
    mid = 0
    for i in app:
        if i != 0:
            mid += W * math.log(i)
        else:
            mid += 0
    BLEU = BP * math.exp(mid)
    return BLEU

def b_bleu(pre_batch, rea_batch, N):
    '''

    Parameters
    ----------
    pre_batch : TYPE = list
       pre_batch = [tensor1, tensor2, ...], is the predict id of samples.
    rea_batch : TYPE = list
       rea_batch = [tensor1, tensor2, ...], is the real id of samples.

    Returns
    -------
    blue of this batch samples
    '''
    bleu = 0
    assert len(pre_batch) == len(rea_batch)
    for i in range(len(pre_batch)):
        bleu += s_bleu(pre_batch[i],rea_batch[i], N)
    return bleu/(len(pre_batch))




def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]    ## 这里的-1表示最大的top_k个元素中最小的那个
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  ## 安全性操作，保证sorted_indices_to_remove的第0个token是False，这样就至少有一个元素不会被移除
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]   ## 操作会留下sorted_indices_to_remove中True的元素（要移除的元素）
        logits[indices_to_remove] = filter_value
    return logits


##  双重终止符或许能够让机器人更加强调个体性， 单重终止符号或许能让机器人强调连贯性。
def talk(model, talk_file_path,  tokenizer, tok = 8, top = 0, maxlong = 20, repetition_penalty = 1.25):
    
    if not os.path.exists(talk_file_path):
        os.mkdir(talk_file_path)
    samples_file = open(os.path.join(talk_file_path, 'talk_record.txt'),'a',encoding='utf8')
    samples_file.write(repr(datetime.datetime.now())+'\n')
    samples_file.write('we use model is %s:%s \n'%(model_file.split('/')[-2],model_file.split('/')[-1]))
    temperature = 1
    
    history = []
    print('file will be record at %s '%talk_file_path)
    print('we use model is %s:%s'%(model_file.split('/')[-2],model_file.split('/')[-1]))
    print('开始和chatbot聊天，输入 quit 以退出')
    
    flag = 0
    while True:
        text = input("user:")
        if text == "quit" and flag == 0:
            break
        elif text == "quit" and flag > 0:
            samples_file.close()
        samples_file.write("user:{}\n".format(text))
        #pdb.set_trace()
        # pdb.set_trace()
        encoder1 = tokenizer.encode(text)
        history.append(encoder1)
        input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        #input_ids = []
        for history_id, history_utr in enumerate(history[-5:]):
            input_ids.extend(history_utr)
            input_ids.append(tokenizer.sep_token_id)
        curr_input_tensor = torch.tensor(input_ids).long().reshape(1,-1).to(device)
        
        #pdb.set_trace()
        # curr2 = curr_input_tensor.reshape(1,-1)
        
        generated = []
        # 最多生成max_len个token
        for p in range(maxlong):
            
            outputs = model(curr_input_tensor)
            next_token_logits = outputs[0][0][-1,:]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(generated):
                next_token_logits[id] /= repetition_penalty
            next_token_logits = next_token_logits / temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[tokenizer.unk_token_id] = -float('Inf')
            next_token_logits[tokenizer.pad_token_id] = -float('Inf')
            if p==0 and torch.argmax(next_token_logits) == 102:
                next_token_logits[tokenizer.sep_token_id] = -float('inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=tok, top_p=top)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            generated.append(next_token.item())
            curr_input_tensor = torch.cat((curr_input_tensor, next_token.unsqueeze(0)), dim=1).reshape(1,-1)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))
        history.append(generated)
        text = tokenizer.convert_ids_to_tokens(generated)
        print("chatbot:" + "".join(text))
        samples_file.write("chatbot:{}\n".format("".join(text)))
            
def patch_src(src):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:]
    return trg, gold



totbit = 1999
long = len(corpus)
sur = split(length = long, sublength = long//totbit)

bs = 16
test_corpus = corpus[sur[-2:][0][0]:]
tensor_cor = Lccd2Tensor(test_corpus, flag =0, totbit = totbit, tokenizer = tokenizer,maxlong=256)
testloader = DataLoader(dataset = tensor_cor, batch_size = bs, collate_fn = mycollate_fn)


print("模型参数总量为： %d"%model.num_parameters())
BLEU2 = []
BLEU4 = []
R_LCS = []
P_LCS = []
F_LCS = []


# for i,inp in enumerate(testloader):
#     inp = inp.to(device)
#     src = patch_src(inp)
#     _, gold = patch_trg(inp)
    
#     output = model(src)[0]   
#     output = output[:,:-1,:]                                                                                  
    
#     predict = []
#     presentence = []
#     realsentence = []
#     for sen in range(output.shape[0]):
#         feature = output[sen,:,:]
#         prelabel = torch.argmax(feature, dim = 1)
#         predict.append(prelabel)
#         presen = tensor2string(prelabel)
#         presentence.append(presen)
#     predict = torch.stack(predict, dim = 0)
    
#     for sen in range(gold.shape[0]):
#         senid = gold[sen]
#         realsen = tensor2string(senid)
#         realsentence.append(realsen)
#     realid = gold
#     bleu2 = b_bleu(predict, realid, 2)
#     bleu4 = b_bleu(predict, realid, 4)
#     r_lcs,p_lcs,f_lcs = batch_rouge_l(predict, realid)
#     BLEU2.append(bleu2)
#     BLEU4.append(bleu4)
#     R_LCS.append(r_lcs)
#     P_LCS.append(p_lcs)
#     F_LCS.append(f_lcs)
#     if i%10 == 0:
#         print("i:",i,'; bleu2: ',bleu2, '; bleu4: ', bleu4, '; r_lcs: ',r_lcs, '; p_lcs: ',p_lcs, '; f_lcs: ',f_lcs)
    
# print("测试集平均分数, bleu2: %.4f, bleu4: %.4f, r_lcs: %.4f, p_lcs: %.4f, f_lcs: %.4f"%
#       tuple(map(lambda x:sum(x)/len(x), [BLEU2,BLEU4,R_LCS,P_LCS,F_LCS])))


# with open("/Users/shizi/Desktop/project/baseline/GPT2/bleu2.txt",'w', encoding = 'utf8') as f:
#     f.write(repr(BLEU2))
# with open("/Users/shizi/Desktop/project/baseline/GPT2/bleu4.txt",'w', encoding = 'utf8') as f:
#     f.write(repr(BLEU4))
# with open("/Users/shizi/Desktop/project/baseline/GPT2/r_lcs.txt",'w', encoding = 'utf8') as f:
#     f.write(repr(R_LCS))
# with open("/Users/shizi/Desktop/project/baseline/GPT2/p_lcs.txt",'w', encoding = 'utf8') as f:
#     f.write(repr(P_LCS))
# with open("/Users/shizi/Desktop/project/baseline/GPT2/f_lcs.txt",'w', encoding = 'utf8') as f:
#     f.write(repr(F_LCS))

talk(model, os.path.join('/Users/shizi/Desktop/project/S2S/talk_record',model_file.split('/')[-1]), tokenizer)
