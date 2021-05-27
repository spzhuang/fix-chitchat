#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:51:51 2020

@author: shizi
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default="./info.txt")
parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                    help='对话模型输出路径')
args = parser.parse_args()
with open(args.model_path, "r", encoding="utf8") as f:
    content = f.read()
    
print(content)
