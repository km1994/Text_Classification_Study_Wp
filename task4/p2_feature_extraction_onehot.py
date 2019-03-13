# coding: utf-8

import sys
from collections import Counter
import jieba
import numpy as np
import tensorflow.contrib.keras as kr

'''
    工具包 begin
'''
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

'''
    工具包 end
'''

'''
    文件读写 begin
'''
#读取文件数据
def read_file(filename,demo_flag = False):
    '''
    读取文件数据
    :param filename:    String 文件名称包含路径
    :param demo_flag:   String True 只读取 1000 样本数据，Fasle 读取全部数据
    :return:
        contents:   list    内容列表
        labels:     list    内容标签
    '''
    contents, labels = [], []
    contents_num = 0
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(''.join(list(native_content(content))))
                    labels.append(native_content(label))
                contents_num = contents_num + 1
                if demo_flag and contents_num == 1000:
                    break
            except:
                pass
    return contents, labels

# 数据写入
def write_file(filename,contents,labels,is_num = False):
    '''
    数据写入
    :param filename:    String  写入文件名称
    :param contents:    List    写入的数据列表
    :param labels:      Lisit   写入的标签数据
    :param is_num:      True 写入数值数据，False 写入字符串数据
    :return:
    '''
    with open_file(filename,mode='w') as f:
        for (content,label) in zip(contents,labels):
            f.write(label)
            f.write("\t")
            if is_num:
                f.write(str(content).replace("[","").replace("]",""))
            else:
                f.write(" ".join(content))
            f.write("\n")

def write_feature_dict_file(filename,contents):
    with open_file(filename,mode='w') as f:
        for content in contents:
            f.write(content)
            f.write("\n")
'''
    文件读写 end
'''

'''
    文本分词 begin
'''
# 数据分词，及停用词清除
def jieba_cut_word(data_list,stopword_path,cut_word_to_str_flag = True):
    '''
    数据分词，及停用词清除
    :param data_list:               list  需要处理的数据列表
    :param stopword_path:           String 停用词文件
    :param cut_word_to_str_flag:    True 切分后的句子转化为字符串， False 切分后的句子转化为列表
    :return:
        docs_list   List    处理后的数据列表
    '''
    # 1.读取停用词文件
    with open_file(stopword_path) as f_stop:
        try:
            f_stop_text = f_stop.read()
        finally:
            f_stop.close()
    # 停用词清除
    f_stop_seg_list = f_stop_text.split('\n')

    # 2.文本分词处理，并进行清除停用词处理
    docs_list = []
    #print("data_list:{0}".format(data_list[0:2]))
    for line in data_list:
        seg_list = jieba.cut(line, cut_all=False)
        word_list = list(seg_list)
        mywordlist = []
        for myword in word_list:
            if not (myword in f_stop_seg_list):
                mywordlist.append(myword)

        if cut_word_to_str_flag:
            mywordlist =" ".join(mywordlist)

        docs_list.append(mywordlist)

    return docs_list

'''
    文本分词 end
'''

'''
    特征提取 begin
'''
# 统计词频
def count_words(train_data_list):
    '''
    统计词频
    :param train_data_list:  训练集
    :return: all_words_list:  词频
    '''
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list.split(" "):
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # 降序排序（key函数）
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list = list(zip(*all_words_tuple_list))[0]
    return all_words_list

# 特征选择
# 仅选取词频坐高的1000个特征词（维度），并剔除数字与停用词。
def words_dict(all_words_list,deleteN = 1000):
    '''
    特征选择 : 仅选取词频坐高的1000个特征词（维度），并剔除数字与停用词。
    :param all_words_list:  list   词频列表
    :param deleteN:         int    选取特征数
    :return:
        feature_words：      list   特征列表
    '''
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n > 1000:    # 最多取1000个维度
            break
        if not all_words_list[t].isdigit() and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words

'''
    特征提取 end
'''

'''
    特征词转化为 One-hot 矩阵 begin
'''
# 用选取的特征词构建0-1矩阵
def text_features(train_data_list, test_data_list, feature_words):
    '''
    # 用选取的特征词构建0-1矩阵
    # 对训练数据集train_data_list中每篇切完词之后的文档构建特征向量（
    # 由上述1000个特征词组成），若出现则取值为1，否则为0。
    # 于是文章构建出了[90,1000]维度的0-1矩阵。
    :param train_data_list:     list        训练集列表
    :param test_data_list:      list        测试集列表
    :param feature_words:       list        特征词列表
    :return:
    '''
    def text_features(text, feature_words):
        # text = train_data_list[0]
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    # 0,1的矩阵（1000列-维度）
    train_feature_list = [text_features(text.split(" "), feature_words) for text in train_data_list]
    test_feature_list = [text_features(text.split(" "), feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

'''
    特征词转化为 One-hot 矩阵 end
'''

if __name__ == "__main__":
    #文件读取
    train_file_path = "../resource/THUCNews_ch/t1_cut_words_cnews.train.txt"
    train_list, train_labels = read_file(train_file_path,demo_flag = False)
    print("train_list:{0},train_labels{1}".format(len(train_list),len(train_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:1], train_labels[0:1]))

    test_file_path = "../resource/THUCNews_ch/t1_cut_words_cnews.test.txt"
    test_list, test_labels = read_file(test_file_path, demo_flag=False)
    print("test_list:{0},test_labels{1}".format(len(test_list), len(test_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:1], train_labels[0:1]))

    # 统计词频
    all_words_list = count_words(train_list)
    # print(f"all_words_list:{all_words_list}")

    # 选取1000个特征
    deleteN = 5000
    feature_words = words_dict(all_words_list, deleteN)
    # print("feature_words：{0}".format(feature_words))

    # 计算特征向量
    train_feature_list, test_feature_list = text_features(train_list, test_list, feature_words)
    # print("train_feature_list:{0}".format(train_feature_list))

    feature_dict_filename = "../resource/THUCNews_ch/t2_feature_words_cnews.txt"
    write_feature_dict_file(feature_dict_filename, feature_words)

    train_feature_filename = "../resource/THUCNews_ch/t2_text_feature_cnews.train.txt"
    write_file(train_feature_filename, train_feature_list, train_labels, is_num=True)

    test_feature_filename = "../resource/THUCNews_ch/t2_text_feature_cnews.test.txt"
    write_file(test_feature_filename, test_feature_list, test_labels, is_num=True)


    # del cut_words_train_filename, train_docs_list, train_labels, cut_words_test_filename, test_docs_list, test_labels
    #
