# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

# 如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码
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

# 常用文件操作，可在python2和python3间切换.
def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

# 读取文件数据
def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

'''
    文本分词 begin
'''
# 数据分词，及停用词清除
import jieba
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


# 读取文件数据,并利用结巴分词进行分词
def read_file_to_words(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(''.join(list(native_content(content))))
                    labels.append(native_content(label))
            except:
                pass

    # 分词处理
    stopword_path = "../../resource/stopwords.txt"
    train_docs_list = jieba_cut_word(contents, stopword_path, cut_word_to_str_flag=False)
    return train_docs_list, labels

#根据训练集构建词汇表，存储
def build_vocab_to_words(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    print("------------build_vocab_to_words begin-------------")
    data_train, _ = read_file_to_words(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    print("------------build_vocab_to_words end-------------")

#根据训练集构建词汇表，存储
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    '''
        print("data_train[0:1]:{0}".format(data_train[0:1]))
        print("all_data[0:1]:{0}".format(all_data[0:1]))
        print("counter:{0}".format(counter))
        print("count_pairs:{0}".format(count_pairs))
        print("words[0:1]:{0}".format(words[0:1]))
        output:
            data_train[0:1]:[['马', '晓', '旭', '意',..., '有', '些', '不', '解', '。']]
            all_data[0:1]:['马']
            counter:Counter({'，': 1871208, '的': 1414310, ... , '溟': 1})
            count_pairs:[('，', 1871208), ('的', 1414310), ('。', 822140),..., ('箕', 9), ('柘', 9)]
            words[0:1]:['<PAD>']
    '''

# 读取词汇表
def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

# 读取分类目录，固定
def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    '''
        print("categories:{0}".format(categories))
        print("cat_to_id:{0}".format(cat_to_id))
        output: 
            categories:['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
            cat_to_id:{'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
    '''
    return categories, cat_to_id

# 将id表示的内容转换为文字
def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

# 将文件转换为id表示
def process_file_to_words(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file_to_words(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

# 生成批次数据
def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
