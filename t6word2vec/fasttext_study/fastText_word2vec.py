# encoding = utf8
'''
    @Author: King
    @Date: 2019.03.16
    @Purpose: 自然语言处理基础知识学习
    @Introduction:  自然语言处理基础知识学习
    @Datasets: THUCNews 情感分析数据集
    @Link : 
    @Reference : 
'''
import fasttext

# Skipgram model
model = fasttext.skipgram('../../resource/THUCNews_ch/t1_cut_words_cnews.train.txt', 'model')
print(model.words) # list of words in dictionary

# CBOW model
model = fasttext.cbow('../../resource/THUCNews_ch/t1_cut_words_cnews.train.txt', 'model')
print(model.words) # list of words in dictionary