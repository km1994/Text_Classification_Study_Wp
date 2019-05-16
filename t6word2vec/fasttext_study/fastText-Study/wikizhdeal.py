# _*_ coding:utf-8 _*_

'''
@Author: King
@Date: 2019.03.13
@Purpose: 保留文件中不含英文字符和数字的行
'''

import codecs
import re
from os import path
from random import shuffle

# 随机打乱字符串顺序
data_dir = 'resource/'

def shuffle_string(string):
    string=list(string)
    shuffle(string)
    return " ".join(string)

symbols=["【","==","===","====","*","#"] # 这个符号可添加
path.join(data_dir, 'test.txt')
with codecs.open(path.join(data_dir, 'wiki.txt'),"r","utf-8") as f1,\
codecs.open(path.join(data_dir, 'wiki_positive.txt'),"w","utf-8") as f2,\
codecs.open(path.join(data_dir, 'wiki_negative.txt'),"w","utf-8") as f3,\
codecs.open(path.join(data_dir, 'train.txt'),"w","utf-8") as f4,\
codecs.open(path.join(data_dir, 'valid.txt'),"w","utf-8") as f5,\
codecs.open(path.join(data_dir, 'test.txt'),"w","utf-8") as f6:
    data=[]
    for line in f1:
        i = 0
        line=line.strip()
        if len(line) > 15:
            for symbol in symbols:
                if symbol not in line:
                    i +=1
            if i == len(symbols):
                #if not re.match(r'[+-]?\d+$', line) and not re.match(r'[A-Za-z]+',line):
                if not re.findall('.*[0-9].*',line): # 判断 line 中是否存在阿拉伯数字
                    if not re.findall('.*[A-Za-z]>*',line): # 判断 line 中是否存在英文字符
                        sentences=line.split("。") # 根据 句号 进行句子切分
                        for sentence in sentences:
                            if len(sentence)< 50 and len(sentence) > 10: # 控制句子的长度 > 10 and < 50
                                #f2.write("{}\n".format(sentence))
                                sentence=sentence+"。"
                                data.append(sentence)
                            else:
                                words=sentence.split("，") # 如果句子的长度>50，根据 逗号 进行句子切分
                                for word in words:
                                    if len(word) > 10: 
                                        word=word+"。"
                                        #f2.write("{}\n".format(word))
                                        data.append(word)
                                        
    splitindex=int(len(data)*0.5)
    
    positives=data[:splitindex]
    negatives=data[splitindex+1:]
    
    positive_label=[]
    negative_label=[]
    
    for positive in positives:
        #positive=" ".join(i for i in positive)+" _label_1"
        positive="__label__1"+"	"+" ".join(i for i in positive)
        positive_label.append(positive)
        f2.write("{}\n".format(positive))
        
    for negative in negatives:
        negative="__label__0"+"	"+shuffle_string(negative)
        #negative=" ".join(i for i in negative)
        negative_label.append(negative)
        f3.write("{}\n".format(negative))
        
    train_data=positive_label[:int(len(positives)*0.5)]+negative_label[:int(len(negatives)*0.5)]
    shuffle(train_data)
    
    valid_data=positive_label[int(len(positives)*0.5)+1:int(len(positives)*0.8)]+\
    negative_label[int(len(negatives)*0.5)+1:int(len(negatives)*0.8)]
    shuffle(valid_data)
    
    test_data=positive_label[int(len(positives)*0.8)+1:]+negative_label[int(len(negatives)*0.8)+1:]
    shuffle(test_data)
    
    for line in train_data:
        f4.write("{}\n".format(line))
       
    for line in valid_data:
        f5.write("{}\n".format(line))
        
    for line in test_data:
        f6.write("{}\n".format(line))