# _*_ coding:utf-8 _*_

'''
@Author: King
@Date: 2019.03.16
@Purpose: tf.nn.embedding_lookup  学习
@Attention: 本例中的负样本是 shuffle 正样例得到，所以容易形成分类
@算法：embedding
@URL introduction：https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
'''
import tensorflow as tf

params = tf.constant([10,20,30,40])
ids1 = tf.constant([0,1,2,3])
ids2 = tf.constant([1,1,3])

params31 = tf.constant([1,2])
params32 = tf.constant([10,20])
ids3 = tf.constant([2,0,2,1,2,3])

# 创建session
session = tf.Session()
session.run(tf.global_variables_initializer())
with session.as_default():
    print("tf.nn.embedding_lookup(params,ids1).eval():{0}".format(tf.nn.embedding_lookup(params,ids1).eval()))
    print("tf.nn.embedding_lookup(params,ids2).eval():{0}".format(tf.nn.embedding_lookup(params, ids2).eval()))
    result3 = tf.nn.embedding_lookup([params31, params32], ids3)
    print("tf.nn.embedding_lookup([params31, params32], ids3):{0}".format(result3.eval()))

    '''
        output:
            tf.nn.embedding_lookup(params,ids1).eval():[10 20 30 40]
            tf.nn.embedding_lookup(params,ids2).eval():[20 20 40]
            tf.nn.embedding_lookup([params31, params32], ids3):[ 2  1  2 10  2 20]
            
        introduction:
        
            The third output:
                index 0 corresponds to the first element of the first tensor: 1
                index 1 corresponds to the first element of the second tensor: 10
                index 2 corresponds to the second element of the first tensor: 2  
                index 3 corresponds to the second element of the second tensor: 20
    '''