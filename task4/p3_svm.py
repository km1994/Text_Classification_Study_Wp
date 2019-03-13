# coding: utf-8

import sys
from collections import Counter
import jieba
import numpy as np

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
def read_file(filename,is_text = True,split_sign=''):
    '''
    读取文件数据
    :param filename:    String 文件名称包含路径
    :param demo_flag:   String True 只读取 1000 样本数据，Fasle 读取全部数据
    :return:
        contents:   list    内容列表
        labels:     list    内容标签
    '''
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    if is_text:
                        contents.append(''.join(list(native_content(content))))
                    else:
                        content = list(map(float,str(native_content(content)).split(split_sign)))
                        contents.append(content)
                    labels.append(native_content(label))
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

# 数据写入，只写入数据，不写入标签数据
def write_feature_dict_file(filename,contents):
    '''
    数据写入，只写入数据，不写入标签数据
    :param filename:    文件名称
    :param contents:    数据
    :return:
    '''
    with open_file(filename,mode='w') as f:
        for content in contents:
            f.write(content)
            f.write("\n")
'''
    文件读写 end
'''

'''
    模型创建与测试 begin
'''
'''
    SVM 分类器创建 begin
'''
from sklearn.svm import SVC
def createSVM(train_x,train_y):
    # 训练朴素贝叶斯
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    svc_model = SVC()
    print('start fit.')
    svc_model.fit(train_x, train_y)
    print('end fit.')
    return svc_model

'''
    SVM 分类器创建 end
'''
'''
    SVM 分类器测试 begin
'''
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
def test_SVM(gs,test_x,test_y):
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    pred = gs.predict(test_x)
    print(pred.shape, test_y.shape, test_x.shape)
    report = classification_report(test_y, pred)
    print(report)

    mat = confusion_matrix(test_y, pred)
    print(mat)

    acc = np.sum(pred == test_y) / len(test_y)
    print('acc', acc)
'''
    SVM 分类器测试 end
'''

'''
    模型创建与测试 end
'''

if __name__ == "__main__":
    #文件读取
    onehot_train_file_path = "../resource/THUCNews_ch/t2_onehot_text_feature_cnews.train.txt"
    onehot_train_list, onehot_train_labels = read_file(onehot_train_file_path, is_text = False, split_sign=',')
    # print("train_list:{0},train_labels{1}".format(len(train_list),len(train_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:3], train_labels[0:3]))

    onehot_test_file_path = "../resource/THUCNews_ch/t2_onehot_text_feature_cnews.test.txt"
    onehot_test_list, onehot_test_labels = read_file(onehot_test_file_path, is_text = False, split_sign=',')
    # print("test_list:{0},test_labels{1}".format(len(test_list), len(test_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:1], train_labels[0:1]))

    onehot_gs = createSVM(onehot_train_list, onehot_train_labels)
    test_SVM(onehot_gs, onehot_test_list, onehot_test_labels)
    print("--------------------One-HOT SVM-----------------------")
    '''
        output:
                      (1000,) (1000,) (1000, 1000)
                     precision    recall  f1-score   support
    
             体育       0.97      1.00      0.99       974
             医疗       0.00      0.00      0.00        14
             建设       0.00      0.00      0.00        12
    
            avg / total       0.95      0.97      0.96      1000
            
            [[974   0   0]
             [ 14   0   0]
             [ 12   0   0]]
            acc 0.974
    '''

    print("--------------------TF-IDF SVM-----------------------")
    # 文件读取
    tfidf_train_file_path = "../resource/THUCNews_ch/t2_tfidf_text_feature_cnews.train.txt"
    tfidf_train_list, tfidf_train_labels = read_file(tfidf_train_file_path, is_text=False, split_sign=',')
    # print("train_list:{0},train_labels{1}".format(len(train_list),len(train_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:3], train_labels[0:3]))

    tfidf_test_file_path = "../resource/THUCNews_ch/t2_tfidf_text_feature_cnews.test.txt"
    tfidf_test_list, tfidf_test_labels = read_file(tfidf_test_file_path, is_text=False, split_sign=',')
    # print("test_list:{0},test_labels{1}".format(len(test_list), len(test_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:1], train_labels[0:1]))

    tfidf_gs = createSVM(tfidf_train_list, tfidf_train_labels)
    test_SVM(tfidf_gs, tfidf_test_list, tfidf_test_labels)

    '''
        output:
             (1000,) (1000,) (1000, 1000)
                precision    recall  f1-score   support

         体育       0.98      1.00      0.99       978
         医疗       0.00      0.00      0.00         9
         建设       0.00      0.00      0.00        13

    avg / total       0.96      0.98      0.97      1000
    
    [[978   0   0]
     [  9   0   0]
     [ 13   0   0]]
    acc 0.978
    '''

    print("--------------------mutual_info SVM-----------------------")
    # 文件读取
    mutual_info_train_file_path = "../resource/THUCNews_ch/t2_mutual_info_text_feature_cnews.train.txt"
    mutual_info_train_list, mutual_info_train_labels = read_file(mutual_info_train_file_path, is_text=False, split_sign=',')
    # print("train_list:{0},train_labels{1}".format(len(train_list),len(train_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:3], train_labels[0:3]))

    mutual_info_test_file_path = "../resource/THUCNews_ch/t2_mutual_info_text_feature_cnews.test.txt"
    mutual_info_test_list, mutual_info_test_labels = read_file(mutual_info_test_file_path, is_text=False, split_sign=',')
    # print("test_list:{0},test_labels{1}".format(len(test_list), len(test_labels)))
    # print("train_list:{0},train_labels{1}".format(train_list[0:1], train_labels[0:1]))

    mutual_info_gs = createSVM(mutual_info_train_list, mutual_info_train_labels)
    test_SVM(mutual_info_gs, mutual_info_test_list, mutual_info_test_labels)

    '''
        output:
            (1000,) (1000,) (1000, 1000)
                    precision    recall  f1-score   support
    
             体育       0.98      1.00      0.99       984
             医疗       0.00      0.00      0.00         7
             建设       0.00      0.00      0.00         9

        avg / total       0.97      0.98      0.98      1000
        
        [[984   0   0]
         [  7   0   0]
         [  9   0   0]]
        acc 0.984
    '''

