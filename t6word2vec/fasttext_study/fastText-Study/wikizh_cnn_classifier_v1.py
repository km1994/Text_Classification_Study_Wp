# _*_ coding:utf-8 _*_

'''
@Author: King
@Date: 2019.03.13
@Purpose: 处理wikizh文本的二分类问题，判断语句是否通顺
@Attention: 本例中的负样本是 shuffle 正样例得到，所以容易形成分类
@算法：CNN
@本例是二分类问题
'''

import codecs
from os import path
data_dir = 'resource/'


train_data_name=path.join(data_dir, 'train.txt')
test_data_name=path.join(data_dir, 'test.txt')

x_train=[]
x_test=[]
y_train=[]
y_test=[]

x_train_positive=0
x_train_negative=0
x_test_positive=0
x_test_negative=0

with codecs.open(train_data_name,"r","utf-8") as f1:
    for line in f1:
        words=line.strip().split("\t")
        if words[0] == "__label__1":
            y_train.append([0,1]) # [0,1] 表示正样例
            x_train_positive += 1
        else:
            y_train.append([1,0]) # [1,0] 表示负样例
            x_train_negative += 1
        x_train.append(words[1])

with codecs.open(test_data_name,"r","utf-8") as f2:
    for line in f2:
        words=line.strip().split("\t")
        if words[0] == "__label__1":
            y_test.append([0,1])
            x_test_positive += 1
        else:
            y_test.append([1,0])
            x_test_negative += 1
        x_test.append(words[1])
        
print("#----------------------------------------------------------#")
print("训练集总数：{}".format(len(x_train)))
print("训练集中正样本个数：{}".format(x_train_positive))
print("训练集中负样本个数：{}".format(x_train_negative))
print("测试集总数：{}".format(len(x_test)))
print("测试集中正样本个数：{}".format(x_test_positive))
print("测试集中负样本个数：{}".format(x_test_negative))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("将输入文本转换成 index - word 对应关系，并输出词汇表")
x_text=x_train+x_test # 总输入文本
y_labels=y_train+y_test

from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import collections

max_document_length=200
min_frequency=1


vocab = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency, tokenizer_fn=list)
x = np.array(list(vocab.fit_transform(x_text)))
vocab_dict = collections.OrderedDict(vocab.vocabulary_._mapping)



with codecs.open(path.join(data_dir, 'vocabulary.txt'),"w","utf-8") as f:
    for key,value in vocab_dict.items():
        f.write("{} {}\n".format(key,value))
        
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("数据混洗")
np.random.seed(10)
y=np.array(y_labels)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

test_sample_percentage=0.2
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("读取預训练词向量矩阵")


embedding_index={}

with codecs.open(path.join(data_dir, 'sgns.wiki.word/sgns.wiki.word'),"r","utf-8") as f:
    #for line in f:
    #    if len(line.strip().split(" "))==2:
    #        nwords=int(line.strip().split(" ")[0])
    #        ndims=int(line.strip().split(" ")[1])
    #    else:
    #        values=line.split()
    #        words=values[0]
    #        coefs=np.asarray(values[1:],dtype="float32")
    #        embedding_index[word]=coefs
    line=f.readline()
    nwords=int(line.strip().split(" ")[0])
    ndims=int(line.strip().split(" ")[1])
    for line in f:
        values=line.split()
        words=values[0]
        coefs=np.asarray(values[1:],dtype="float32")
        embedding_index[words]=coefs
        
print("預训练模型中Token总数：{} = {}".format(nwords,len(embedding_index)))
print("預训练模型的维度：{}".format(ndims))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("将vocabulary中的 index-word 对应关系映射到 index-word vector形式")

embedding_matrix=[]
notfoundword=0

for word in vocab_dict.keys():
    if word in embedding_index.keys():
        embedding_matrix.append(embedding_index[word])
    else:
        notfoundword += 1
        embedding_matrix.append(np.random.uniform(-1,1,size=ndims))
        
embedding_matrix=np.array(embedding_matrix,dtype=np.float32) # 必须使用 np.float32
print("词汇表中未找到单词个数：{}".format(notfoundword))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("构建CNN模型.................")
print("Embedding layer --- Conv1D layer --- Dense layer --- Dense layer")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

max_sentence_length=200
embedding_dims=ndims
input_length=max_sentence_length
batch_size = 64
filters = 250
kernel_size = 3
hidden_dims = 250
dropout=0.5
num_classes=2
epochs = 2

model = Sequential()
model.add(Embedding(len(vocab_dict),
                    embedding_dims,
                    weights=[embedding_matrix],
                    input_length=max_sentence_length,
                    trainable=False))
model.add(Dropout(dropout))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("编译模型")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("模型拟合")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("模型评估")
# 训练得分和准确度

score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)

print("#---------------------------------------------------#")
print("预测得分:{}".format(score))
print("预测准确率:{}".format(acc))
print("#---------------------------------------------------#")
print("\n")

# 模型预测

predictions=model.predict(x_test)

print("#---------------------------------------------------#")
print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
print(predictions)
print("#---------------------------------------------------#")
print("\n")

# 模型预测类别

predict_class=model.predict_classes(x_test)

print("#---------------------------------------------------#")
print("测试集的预测类别")
print(predict_class)
print("#---------------------------------------------------#")
print("\n")

# 模型保存

model.save(path.join(data_dir, 'wikizh_cnn.h5'))

print("#---------------------------------------------------#")
print("保存模型")
print("#---------------------------------------------------#")
print("\n")

# 模型总结

print("#---------------------------------------------------#")
print("输出模型总结")
print(model.summary())
print("#---------------------------------------------------#")
print("\n")

# 模型的配置文件

config=model.get_config()

print("#---------------------------------------------------#")
print("输出模型配置信息")
print(config)
print("#---------------------------------------------------#")
print("\n")



    
