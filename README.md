# DataWhale_Study_Wp 
[DataWhale_Study_Wp](https://github.com/km1994/DataWhale_Study_Wp)
## Task 1 数据集下载和探索
1. 数据集
数据集：中、英文数据集各一份

中文数据集：THUCNews
THUCNews数据子集：https://pan.baidu.com/s/1hugrfRu 密码：qfud
英文数据集： [IMDB数据集 Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/)

2. IMDB数据集下载和探索
参考
[TensorFlow官方教程：影评文本分类  |  TensorFlow](https://tensorflow.google.cn/tutorials/keras/basic_text_classification)

[科赛 - Kesci.com](https://www.kesci.com/home/project/5b6c05409889570010ccce90)


3. THUCNews数据集下载和探索
参考：
[博客中的数据集部分和预处理部分：CNN字符级中文文本分类-基于TensorFlow实现 - 一蓑烟雨 - CSDN博客](https://blog.csdn.net/u011439796/article/details/77692621)

参考代码： 
[text-classification-cnn-rnn/cnews_loader.py at mas...](https://github.com/gaussic/text-classification-cnn-rnn/blob/master/data/cnews_loader.py)


4. 学习召回率、准确率、ROC曲线、AUC、PR曲线这些基本概念
参考1：
[机器学习之类别不平衡问题 (2) —— ROC和PR曲线_慕课手记](https://www.imooc.com/article/48072)


## Task2 特征提取 

1. 基本文本处理技能
1.1 分词的概念（分词的正向最大、逆向最大、双向最大匹配法）；
1.2 词、字符频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）
2. 语言模型
2.1 语言模型中unigram、bigram、trigram的概念；
2.2 unigram、bigram频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）
3. 文本矩阵化：要求采用词袋模型且是词级别的矩阵化
步骤有：
分词（可采用结巴分词来进行分词操作，其他库也可以）；去停用词；构造词表。
每篇文档的向量化。
4. 参考
结巴分词介绍和使用：
[GitHub - fxsjy/jieba: 结巴中文分词](https://github.com/fxsjy/jieba)


## Task3 特征选择  
1. TF-IDF原理。
2. 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）
3. 互信息的原理。
4. 使用第二步生成的特征矩阵，利用互信息进行特征筛选。
5. 参考
[文本挖掘预处理之TF-IDF：文本挖掘预处理之TF-IDF - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/6693230.html)
[使用不同的方法计算TF-IDF值：使用不同的方法计算TF-IDF值 - 简书](https://www.jianshu.com/p/f3b92124cd2b)
[sklearn-点互信息和互信息：sklearn：点互信息和互信息 - 专注计算机体系结构 - CSDN博客](https://blog.csdn.net/u013710265/article/details/72848755)
[如何进行特征选择（理论篇）机器学习你会遇到的“坑”：如何进行特征选择（理论篇）机器学习你会遇到的“坑” ](https://baijiahao.baidu.com/s?id=1604074325918456186&wfr=spider&for=pc)


## Task4 传统机器学习 

1. 朴素贝叶斯的原理
2. 利用朴素贝叶斯模型进行文本分类
3. SVM的原理
4. 利用SVM模型进行文本分类
5. pLSA、共轭先验分布；LDA主题模型原理
6. 使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类
7. 参考
[朴素贝叶斯1：sklearn：朴素贝叶斯（naïve beyes） - 专注计算机体系结构 - CSDN博客 ](https://blog.csdn.net/u013710265/article/details/72780520)

LDA数学八卦
lda2：
[用LDA处理文本(Python) - 专注计算机体系结构 - CSDN博客 ](https://blog.csdn.net/u013710265/article/details/73480332)

[合并特征：Python：合并两个numpy矩阵 - 专注计算机体系结构 - CSDN博客](https://blog.csdn.net/u013710265/article/details/72848564)


## Task6 简单神经网络  

1. 文本表示：从one-hot到word2vec。
1.1 词袋模型：离散、高维、稀疏。
1.2 分布式表示：连续、低维、稠密。word2vec词向量原理并实践，用来表示文本。

2. 走进FastText
2.1 FastText的原理。
2.2 利用FastText模型进行文本分类。

3. 参考
word2vec1：
[word2vec 中的数学原理详解（一）目录和前言 - peghoty - CSDN博客](https://blog.csdn.net/itplus/article/details/37969519)
word2vec2：
 [word2vec原理推导与代码分析-码农场](http://www.hankcs.com/nlp/word2vec.html)

[word2vec中的数学原理详解（四）基于 Hierarchical Softmax 的模型：word2vec 中的数学原理详解（四）基于 Hierarchical Softmax 的模型 - ...](https://github.com/facebookresearch/fastText#building-fasttext-for-python)

fasttext1：
[GitHub - facebookresearch/fastText: Library for fa...](https://github.com/facebookresearch/fastText#building-fasttext-for-python)

fasttext2：
[GitHub - salestock/fastText.py: A Python interface...](https://github.com/salestock/fastText.py)

fasttext3 其中的参考：
[fastText源码分析以及使用 — 相依](https://jepsonwong.github.io/2018/05/02/fastText/)
[A Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.44849&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)



## Task7 卷积神经网络 
 
1. 卷积运算的定义、动机（稀疏权重、参数共享、等变表示）。一维卷积运算和二维卷积运算。
2. 反卷积(tf.nn.conv2d_transpose)
3. 池化运算的定义、种类（最大池化、平均池化等）、动机。
4. Text-CNN的原理。
5. 利用Text-CNN模型来进行文本分类。

参考：

[卷积有多少种？一文读懂深度学习中的各种卷积：卷积有多少种？一文读懂深度学习中的各种卷积 - 知乎](https://zhuanlan.zhihu.com/p/57575810)
源码往期资料中有涉及。

## Task8 循环神经网络  
 
1. RNN的结构。循环神经网络的提出背景、优缺点。着重学习RNN的反向传播、RNN出现的问题（梯度问题、长期依赖问题）、BPTT算法。
2. 双向RNN
3. 递归神经网络
4. LSTM、GRU的结构、提出背景、优缺点。
5、针对梯度消失（LSTM等其他门控RNN）、梯度爆炸（梯度截断）的解决方案。
6. Memory Network（自选）
7. Text-RNN的原理。
8. 利用Text-RNN模型来进行文本分类。
9. Recurrent Convolutional Neural Networks（RCNN）原理。
10. 利用RCNN模型来进行文本分类（自选）

参考

一份详细的LSTM和GRU图解：[一份详细的LSTM和GRU图解 -ATYUN](https://www.atyun.com/30234.html)
Tensorflow实战(1): 实现深层循环神经网络：[Tensorflow实战(1): 实现深层循环神经网络 - 知乎](https://zhuanlan.zhihu.com/p/37070414)
lstm：[从LSTM到Seq2Seq-大数据算法](https://x-algo.cn/index.php/2017/01/13/1609/)
RCNN kreas：[GitHub - airalcorn2/Recurrent-Convolutional-Neural...](https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier)
RCNN tf：[GitHub - zhangfazhan/TextRCNN: TextRCNN 文本分类](https://github.com/zhangfazhan/TextRCNN)
RCNN tf (推荐)：[GitHub - roomylee/rcnn-text-classification: Tensor...](https://github.com/roomylee/rcnn-text-classification)

 