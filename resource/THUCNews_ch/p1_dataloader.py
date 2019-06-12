# coding: utf-8

from collections import Counter
import jieba
import numpy as np


'''
    工具包 begin
'''
import sys
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
        return open(filename, mode, encoding='utf-8-sig', errors='ignore')
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

# 获取 label 类别
def unqiLabel(labels):
    id2label = []
    for label in labels:
        if label not in id2label:
            id2label.append(label)
        
    return id2label

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

if __name__ == "__main__":
    #文件读取
    train_file_path = "cnews.train.txt"
    train_list, train_labels = read_file(train_file_path,demo_flag = False)
    print("train_list:{0},train_labels{1}".format(len(train_list),len(train_labels)))

    train_id2label = unqiLabel(train_labels)
    print("train_id2label:{0}".format(train_id2label))

    dev_file_path = "cnews.val.txt"
    dev_list, dev_labels = read_file(dev_file_path,demo_flag = False)
    print("dev_list:{0},dev_labels{1}".format(len(dev_list),len(dev_labels)))

    dev_id2label = unqiLabel(dev_labels)
    print("dev_id2label:{0}".format(dev_id2label))

    test_file_path = "cnews.test.txt"
    test_list, test_labels = read_file(test_file_path,demo_flag = False)
    print("test_list:{0},test_labels{1}".format(len(test_list),len(test_labels)))

    test_id2label = unqiLabel(test_labels)
    print("test_id2label:{0}".format(test_id2label))