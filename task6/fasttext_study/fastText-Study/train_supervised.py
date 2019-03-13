# _*_ coding:utf-8 _*_

'''
@Author: King
@Date: 2019.03.13
@Purpose: 熟悉 fastText 的python接口，进行文本分类的非监督学习
@Install: 需要预先安装 pybind11,安装过程中需要修改 fastText 中的一些源码，可根据
@         报错信息指定 pybind11 中部分.h文件的所在路径
@目前是不支持进行词向量的训练操作
@下载链接：https://github.com/facebookresearch/fastText
@pybind11: https://github.com/pybind/pybind11
'''

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os
import codecs
from os import path
from fastText import train_supervised


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
            
data_dir = 'resource/'

train_data = path.join(data_dir, 'train.txt')
valid_data = path.join(data_dir, 'valid.txt')
test_data=path.join(data_dir, 'test.txt')

# train_supervised uses the same arguments and defaults as the fastText cli

# 监督学习训练的参数调整
# Reference: https://fasttext.cc/docs/en/supervised-tutorial.html
# 1. 文本预处理
# 2. 改变 epoch 的次数，最佳范围 [5-50]
# 3. 改变学习率，最佳范围 [0.1-1.0]
# 4. 使用 word n-gram，最佳范围 [1-5]

model = train_supervised(input=train_data, epoch=50, lr=0.5, wordNgrams=2, dim=100,verbose=2, minCount=1,thread=1)
print_results(*model.test(valid_data))

model = train_supervised(input=train_data, epoch=50, lr=0.5, wordNgrams=2, dim=100,verbose=2, minCount=1,thread=1,loss="softmax")
print_results(*model.test(valid_data))

model = train_supervised(input=train_data, epoch=50, lr=0.5, wordNgrams=2, dim=100,verbose=2, minCount=1,thread=1,loss="softmax")
print_results(*model.test(test_data))


model.save_model(path.join(data_dir, 'wikizhfasttext.bin'))

model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
print_results(*model.test(valid_data))
model.save_model( path.join(data_dir, 'wikizhfasttext.ftz'))

# 加载训练好的模型

from fastText import FastText

model_path = path.join(data_dir, 'wikizhfasttext.bin')
model=FastText.load_model(model_path)

# help(model)
# 1. model.get_dimension() Get the dimension (size) of a lookup vector (hidden layer).
# 2. model.get_input_matrix() Get a copy of the full input matrix of a Model. This only
#    works if the model is not quantized.
# 3. model.get_input_vector() Given an index, get the corresponding vector of the Input Matrix.
# 4. model.get_labels() Get the entire list of labels of the dictionary optionally
#    including the frequency of the individual labels. Unsupervised
#    models use words as labels, which is why get_labels
#    will call and return get_words for this type of
#    model.
# 5. model.get_line() Split a line of text into words and labels. Labels must start with
#    the prefix used to create the model (__label__ by default)
# 6. model.get_output_matrix() Get a copy of the full output matrix of a Model. This only
#    works if the model is not quantized.
# 7. model.get_sentence_vector() Given a string, get a single vector represenation. This function
#    assumes to be given a single line of text. We split words on
#    whitespace (space, newline, tab, vertical tab) and the control
#    characters carriage return, formfeed and the null character.
# 8. model.get_subword_id() Given a subword, return the index (within input matrix) it hashes to.
# 9. model.get_word_id():Given a word, get the word id within the dictionary.
#    Returns -1 if word is not in the dictionary.
# 10. model.get_word_vector() Get the vector representation of word
# 11. model.get_words() Get the entire list of words of the dictionary optionally
#     including the frequency of the individual words. This
#     does not include any subwords. For that please consult
#     the function get_subwords.
# 12. model.is_quantized()
# 13. model.predict() Given a string, get a list of labels and a list of
#     corresponding probabilities. k controls the number
#     of returned labels. A choice of 5, will return the 5
#     most probable labels. By default this returns only
#     the most likely label and probability. threshold filters
#     the returned labels by a threshold on probability. A
#     choice of 0.5 will return labels with at least 0.5
#     probability. k and threshold will be applied together to
#     determine the returned labels.
# 
#     This function assumes to be given
#     a single line of text. We split words on whitespace (space,
#     newline, tab, vertical tab) and the control characters carriage
#     return, formfeed and the null character.
# 
#     If the model is not supervised, this function will throw a ValueError.
# 
#     If given a list of strings, it will return a list of results as usually
#     received for a single line of text.
# 14. model.quantize() Quantize the model reducing the size of the model and
#     it's memory footprint.
# 15. model.save_model() Save the model to the given path
# 16. model.test() Evaluate supervised model using file given by path
# 17. model.test_label() Return the precision and recall score for each label.
#     The returned value is a dictionary, where the key is the label.
#     For example:
#     f.test_label(...)
#     {'__label__italian-cuisine' : {'precision' : 0.7, 'recall' : 0.74}}

# 使用model中的方法进行新文本预测的时候，需要余弦将文本分好词

strings=["化 史 历 城 名 千 是 定 年 古 。 正 城 县 文","这 就 使 得 脂 类 成 为 同 时 具 有 疏 水 性 和 亲 水 性 的 两 性 分 子 ， 这 一 特 点 也 是 磷 脂 能 够 形 成 双 分 子 层 细 胞 膜 的 原 因 。","《 时 代 》 杂 志 将 格 瓦 拉 选 入 二 十 世 纪 百 大 影 响 力 人 物 。"]

for string in strings:
    result=model.predict(string)
    string="".join(i for i in string)
    print("{}".format(string))
    print("类别标签:{}".format(result[0]))
    print("概率：{}".format(result[1]))
