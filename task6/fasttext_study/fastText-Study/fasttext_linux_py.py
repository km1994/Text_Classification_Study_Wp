# _*_ coding:utf-8 _*_

'''
@Author: King
@Date: 2019.03.13
@Purpose: 基于 subprocess 模块调用二进制 fastText 可执行文件
'''

import os
import subprocess
import codecs
from os import path

datapath='resource/'
inputfile="wiki_positive.txt"

# 训练词向量
# 输出文件
# .bin: 二进制文件，用于存储整个 fastText 模型
# .vec: 包含词向量的文本文件，词汇表中的一个单词对应一行

def trainwordvector(method,inputfile,dim=100,minn=2,maxn=5,epoch=5,\
lr=0.05,thread=1):
    '''
    method: skipgram or cbow 
    inputfile: 输入文件名
    dim: 控制向量的大小，维度越多包含的信息越多，默认使用100维，建议范围 100-300
    minn和maxn描述的是子词大小
    minn: 子词的最小尺寸，默认值 2
    maxn: 子词的最大尺寸，默认值 5
    控制子词大小在3-6个字符，不同语言存在差异
    epoch: 控制训练的轮次，数据集越大可以降低，默认值 5
    lr: 学习率，其值越高训练速度快，但是容易过拟合，默认值 0.05，建议范围[0.01,1]
    thread: 控制训练所使用的线程数，默认是1
    '''
    subprocess.Popen("mkdir wordvector_%s"%(method),shell=True)
    subprocess.Popen("fasttext %s -input %s -output wordvector_%s/%s -dim %d\
    -minn %d -maxn %d -epoch %d -lr %.4f -thread %d"%(method,\
    inputfile,method,inputfile.split(".")[0],dim,minn,maxn,epoch,lr,thread),shell=True)
    
# 打印词向量 print-word-vectors

def printwordvectors(wordfilename,inputfile,method):
    '''
    wordfilename: 存储需要打印词向量的单词
    format: 一个单词一行
    method: skipgram or cbow 
    inputfile: 输入文件名
    '''
    with codecs.open(wordfilename,"r","utf-8") as f:
        data=[]
        for line in f:
            data.append(line.strip())
        wordstring=" ".join(word for word in data)
    subprocess.Popen("echo %s | fasttext print-word-vectors wordvector_%s/%s.bin\
    "%(wordstring,method,inputfile.split(".")[0]),shell=True)
    
# 最近邻查询

def nn(nnwordfilename,inputfile,method):
    '''
    nnwordfilename: 需要进行最近邻查找的单词组
    format: 一个单词一行
    method: skipgram or cbow 
    inputfile: 输入文件名
    '''
    with codecs.open(wordfilename,"r","utf-8") as f:
        data=[]
        for line in f:
            data.append(line.strip())
            
    for word in data:
        subprocess.Popen("echo %s | fasttext nn wordvector_%s/%s.bin"%(word,method,\
        inputfile.split(".")[0]),shell=True)
        
# 文本分类
def textclassifier(trainfile,modelname,epoch=25,lr=0.05,wordNgrams=2):
    '''
    trainfile: 训练模型的文件名称
    modelname: 输出模型名称
    epoch: 训练的轮次，标准范围 [5-50]，默认值25
    lr: 学习率 标准范围 [0.1-1.0]，默认值0.05
    wordNgrams: word n-grams，标准范围 [1-5]，默认值2
    '''
    subprocess.Popen("fasttext supervised -input %s -output %s -epoch %d -lr %.4f -wordNgrams %d\
    "%(trainfile,modelname,epoch,lr,wordNgrams),shell=True)
    
# 测试分类器
def testmodel(testfile,modelname):
    '''
    testfile：测试数据集
    format: 一句话表示一行
    modelname：训练的模型名称
    '''
    with codecs.open(testfile,"r","utf-8") as f:
        data=[]
        for line in f:
            line=" ".join(i for i in line)
            data.append(line)
            
    for line in data:
        subprocess.Popen("echo %s | fasttext predict %s"%(line,modelname),shell=True)
        
# 测试模型在验证集和测试集上的效果
# validation.txt：预先分好词的文本
# test.txt: 预先分好词的文本

def test(modelname,validationname,testname):
    '''
    modelname：训练的模型名称
    validationname: 验证集名称
    testname: 测试机名称
    '''
    subprocess.Popen("fasttext test %s %s"%(modelname,validationname),shell=True)
    subprocess.Popen("fasttext test %s %s"%(modelname,testname),shell=True)

