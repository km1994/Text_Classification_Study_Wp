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
from gensim.models import word2vec
import logging

# 主程序
file_path = "../resource/THUCNews_ch/t1_cut_words_cnews.txt"
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(file_path)  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型，默认window=5
print(model)

# 计算两个词的相似度/相关程度
try:
    y1 = model.similarity(u"姚明", u"易建联")
except KeyError:
    y1 = 0
print(u"【姚明】和【易建联】的相似度为：", y1)
print("-----\n")
#
# 计算某个词的相关词列表
y2 = model.most_similar(u"易建联", topn=20)  # 20个最相关的
print(u"和【易建联】最相关的词有：\n")
for item in y2:
    print(item[0], item[1])
print("-----\n")

# 寻找对应关系
print(u"易建联-中国，安东尼-")
y3 = model.most_similar([u'易建联', u'中国'], [u'安东尼'], topn=3)
for item in y3:
    print(item[0], item[1])
print("----\n")

# 寻找不合群的词
y4 = model.doesnt_match(u"易建联 是 优秀的 运动员".split())
print(u"不合群的词：", y4)
print("-----\n")

# 保存模型，以便重用
model.save(u"cnews_vord2vec.model")
# 对应的加载方式
# model_2 =word2vec.Word2Vec.load("text8.model")

# 以一种c语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)
