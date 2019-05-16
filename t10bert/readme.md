## 自然语言处理基础知识学习

### 2、Feature extraction

#### 介绍

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

#### Requrements

* Python (>=3.5) 

* jieba 


#### 理论学习
[结巴分词介绍和使用](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000463&idx=5&sn=10e383eee7b8d55bfc4e1a518720a8ce&chksm=1bbfe7f52cc86ee3f76535506ced784fc15ac60f5fbc71bbfa9be9cbe3138ed0d0cbe22e4e1b&scene=20&xtrack=1#rd)

#### 算法代码链接

[t2FeatureExtraction](t2FeatureExtraction/)

### 3、Feature select 特征选择  

#### 介绍

1. TF-IDF原理。
2. 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）
3. 互信息的原理。
4. 使用第二步生成的特征矩阵，利用互信息进行特征筛选。

#### Requrements

* Python (>=3.5) 

* jieba 

* sklearn

* numpy


#### 理论学习
[TF-IDF原理及实现](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000529&idx=2&sn=e174dad92fbc53f41f66fc009135702a&chksm=1bbfe62b2cc86f3d3913129272f04e57a30dedab89583720ef9a6062c524e3ce8f5aef93611f&scene=20&xtrack=1#rd)

[互信息的原理及实践](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000529&idx=3&sn=c59720b646aa79eabc10280a21c5b287&chksm=1bbfe62b2cc86f3d8b87659e91d1df1eb4df8447e8e804947d649b9384cff8e5e5a71e907e36&scene=20&xtrack=1#rd)

#### 算法代码链接

[TF-IDF](t3FeatureSelect/t1_tdidf_study.py)

[文本矩阵化，使用词袋模型](t3FeatureSelect/t2_vector_study.py)

[互信息](t3FeatureSelect/t2_mutual_info_study.py)


### 4、传统机器学习及文本分类 

#### 介绍

1. 朴素贝叶斯的原理
2. 利用朴素贝叶斯模型进行文本分类
3. SVM的原理
4. 利用SVM模型进行文本分类
5. pLSA、共轭先验分布；LDA主题模型原理
6. 使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类

#### Requrements

* Python (>=3.5) 

* jieba 

* sklearn

* numpy


#### 理论学习

1.[利用朴素贝叶斯模型进行文本分类](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000564&idx=2&sn=692438a44c21a68a5b5da71173d23c50&chksm=1bbfe60e2cc86f18b39de6096be74ae0588fcb1c0fad0ded7fcfdb5021e109d267cc74415ef8&scene=20&xtrack=1#rd)

2.[利用SVM进行文本分类](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000564&idx=3&sn=9bd6174872a28a1ba87f5737fa372a7d&chksm=1bbfe60e2cc86f185c8932c18c9012ff29f40ffe29392ad8bbdb21fed14c49aa279e0c2f021e&scene=20&xtrack=1#rd)

3.[pLSA、共轭先验分布；LDA主题模型原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000564&idx=4&sn=026a3d75885a9035eff02aff79e80291&chksm=1bbfe60e2cc86f1845d77d1e5146156a77a4971925613fbbea202061c1ee460d4e1272d41415&scene=20&xtrack=1#rd)

4.[pLSA、共轭先验分布；LDA主题模型原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000564&idx=5&sn=520fa61ec19590aa576e67d78e9aa4d8&chksm=1bbfe60e2cc86f18ce455b6984cdac7d15d4fd35ef40767af74602fbd7d8be749f1a9db647fe&scene=20&xtrack=1#rd)

#### 算法代码链接

1.[p1_dataloader 文本数据加载](t4NBandSVN/p1_dataloader.py)

2.[p2_feature_extraction_onehot 基于onehot 的特征提取](t4NBandSVN/p2_feature_extraction_onehot.py)

3.[p2_text_features_by_tfidf 基于 tfidf 的特征提取](t4NBandSVN/p2_text_features_by_tfidf.py)

4.[p2_feature_extraction_mutual_info 基于 信息熵 的特征提取](t4NBandSVN/p2_feature_extraction_mutual_info.py)

5.[p3_naive_bayes 利用朴素贝叶斯进行文本分类型](t4NBandSVN/p3_naive_bayes.py)

6.[p3_svm 利用 SVN 进行文本分类型](t4NBandSVN/p3_svm.py)

7.[p4_lda_study LDA主题](t4NBandSVN/p4_lda_study.py)


### 5、神经网络基础

#### 介绍

1. 前馈神经网络、网络层数、输入层、隐藏层、输出层、隐藏单元、激活函数的概念。
2. 感知机相关；利用tensorflow等工具定义简单的几层网络（激活函数sigmoid），递归使用链式法则来实现反向传播。
3. 激活函数的种类以及各自的提出背景、优缺点。（和线性模型对比，线性模型的局限性，去线性化）
4. 深度学习中的正则化（参数范数惩罚：L1正则化、L2正则化；数据集增强；噪声添加；early stop；Dropout层）、正则化的介绍。
5. 深度模型中的优化：参数初始化策略；自适应学习率算法（梯度下降、AdaGrad、RMSProp、Adam；优化算法的选择）；batch norm层（提出背景、解决什么问题、层在训练和测试阶段的计算公式）；layer norm层。

#### Requrements

* Python (>=3.5) 

* numpy


#### 理论学习

1.[前馈神经网络、网络层数、输入层、隐藏层、输出层、隐藏单元、激活函数的概念](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=2&sn=9a89a8cca58dc4226ca710ca2fed4100&chksm=1bbfe1592cc8684f8e036a3ee900b119b545d29ad4eaf57787a325cdb9d51977e41209c0a057&scene=20&xtrack=1#rd)

2.[感知机相关；利用tensorflow等工具定义简单的几层网络（激活函数sigmoid），递归使用链式法则来实现反向传播](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=3&sn=e51617747de265f9f242e602d9fed696&chksm=1bbfe1592cc8684f5003886f490fc8f9521e1cff218991532a560aeb45ea5b9726fd86fca261&scene=20&xtrack=1#rd)

3.[激活函数的种类以及各自的提出背景、优缺点。（和线性模型对比，线性模型的局限性，去线性化）](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=4&sn=762a102bf05f7280156c23efbbb62912&chksm=1bbfe1592cc8684f12f890777744f4ea9736704f2b91f26dae01a9721182eb9dffba2da594a1&scene=20&xtrack=1#rd)

4.[深度学习中的正则化](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000564&idx=5&sn=520fa61ec19590aa576e67d78e9aa4d8&chksm=1bbfe60e2cc86f18ce455b6984cdac7d15d4fd35ef40767af74602fbd7d8be749f1a9db647fe&scene=20&xtrack=1#rd)

5.[深度模型中的优化](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=6&sn=7182dbcb3eaee3c21dbc00c960343c27&chksm=1bbfe1592cc8684f77f8bf4a947cf50055d18eccb7a382ffeca099ab901be121b923521e7456&scene=20&xtrack=1#rd)

6.[深度模型中的优化](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=7&sn=cb24c641d5dc7a889d77c8af5c0f3e3b&chksm=1bbfe1592cc8684fda2b8fd7694304bec2ae22cda451649ff8705cb0d0c1da0d0bcae3af09d9&scene=20&xtrack=1#rd)

#### 算法代码链接


### 6、简单神经网络  

#### 介绍

1. 文本表示：从one-hot到word2vec。
1.1 词袋模型：离散、高维、稀疏。
1.2 分布式表示：连续、低维、稠密。word2vec词向量原理并实践，用来表示文本。

2. 走进FastText
2.1 FastText的原理。
2.2 利用FastText模型进行文本分类。

#### Requrements

* Python (>=3.5) 

* jieba 

* gensim

* fasttext

* pandas

* sklearn


#### 理论学习

1.[简单神经网络](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000893&idx=1&sn=2dfe4a0548c8ab350cb64d0c3822e746&chksm=1bbfe1472cc8685116a9a7248c4ace9b3a775565eb99ef475fcfaad5622ac07bf5c076374832&scene=20&xtrack=1#rd)

2.[文本表示：从one-hot到word2vec](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000893&idx=2&sn=3eed143ba5dd759dc61a496acf2f9496&chksm=1bbfe1472cc86851c56b7e33209a90d6483faa707e66d89cd3da6e48291fabff867c5d45714a&scene=20&xtrack=1#rd)

3.[FastText 学习](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000893&idx=3&sn=40da18ed12866d5e21d6e99c35911afe&chksm=1bbfe1472cc86851a94e88107164466d0fbbee961f4adb93e2c2ae6a1cb6238f2696b1b7d0ac&scene=20&xtrack=1#rd)


#### 算法代码链接

1.[word2vec_study ](t6word2vec/fasttext_study/word2vec_study.py)

2.[fastText_word2vec ](t6word2vec/fasttext_study/fastText_word2vec.py)

3.[fastText_word2vec 基于 tfidf 的特征提取](t6word2vec/fasttext_study/fastText_word2vec.py)

4.[fastText_text_classifier ](t6word2vec/fasttext_study/fastText_text_classifier.py)

5.[classification ](t6word2vec/fasttext_study/classification.py)


### 7、卷积神经网络   t7CNN

#### 介绍

1. 卷积运算的定义、动机（稀疏权重、参数共享、等变表示）。一维卷积运算和二维卷积运算。

2. 反卷积(tf.nn.conv2d_transpose)

3. 池化运算的定义、种类（最大池化、平均池化等）、动机。

4. Text-CNN的原理。

5. 利用Text-CNN模型来进行文本分类。


#### Requrements

* Python (>=3.5) 

* jieba 

* gensim

* fasttext

* pandas

* sklearn

* tensorflow


#### 理论学习

1. [卷积运算的定义、动机](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=2&sn=4eced9e7204274a4e3798cb73a140c72&chksm=1bbfe1f72cc868e1f63262fad39b4f735a6d424c064f6bee755e438b94487bf75b5d41cc02c0&scene=20&xtrack=1&key=fe048f5ad4fa1bcff1ed72e320faab18cb01c02c1a16279c60775734b428e42206e9f5a8f3c402ae96c01259df639ca04206da43e2ab1b42bfaf44bb4068c793df27faa893ea0301a375086e4adfd3b7&ascene=1&uin=MjQ3NDIwNTMxNw%3D%3D&devicetype=Windows+10&version=62060426&lang=zh_CN&pass_ticket=906xy%2Fk9TQJp5jnyekYc89wLa17ODmZRkas9HXdX%2BtYcy0q32NIxLHOhFx519Yxa)

2. [反卷积Deconvolution](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=3&sn=8a787cc0e165fa071ca7a602f16fae17&chksm=1bbfe1f72cc868e1249a3ebe90021e2a6e12c3d8021fcc1877a5390eed36f5a8a6698eb65216&scene=20&xtrack=1#rd)

3. [池化运算的定义](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=4&sn=cebf71790dd7e0e497fa36fa199c368d&chksm=1bbfe1f72cc868e1017d26a996f1eb7602fad1efced62713def3012b8df1b85b6ba46c0ebae8&scene=20&xtrack=1#rd)

4. [Text-CNN的原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=5&sn=1b8943df5effd562df8a0dea8ddf84a9&chksm=1bbfe1f72cc868e1e0e5f71b5b5bcdf59056d25d3056d05fb932fec942239868c2acca724017&scene=20&xtrack=1#rd)

5. [利用Text-CNN模型来进行文本分类](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=6&sn=baf497892301af2011a0f01da1be2136&chksm=1bbfe1f72cc868e17e63766dfa18b0e75de369c1d4cc4478c1a1a58538591e645b88acb924f3&scene=20&xtrack=1#rd)

#### 算法代码链接

1.[利用Text-CNN模型来进行文本分类 ](t7CNN/text-classification-cnn-rnn.py)



### 8、循环神经网络   t8RNN

#### 介绍

1. RNN的结构。循环神经网络的提出背景、优缺点。着重学习RNN的反向传播、RNN出现的问题（梯度问题、长期依赖问题）、BPTT算法。

2. 双向RNN
3. 递归神经网络

4. LSTM、GRU的结构、提出背景、优缺点。

5. 针对梯度消失（LSTM等其他门控RNN）、梯度爆炸（梯度截断）的解决方案。

6. Memory Network

7. Text-RNN的原理

8. 利用Text-RNN模型来进行文本分类

9. Recurrent Convolutional Neural Networks（RCNN）原理

10. 利用RCNN模型来进行文本分类


#### Requrements

* Python (>=3.5) 

* jieba 

* tensorflow


#### 理论学习

1. [RNN的结构。双向RNN](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=2&sn=4357ec1794170bbd148c92a68073b738&chksm=1bbfe0082cc8691eb737f1cddd1383735a579b2974034803f78e2156038182df1cdcfaedfa93&scene=20&xtrack=1#rd)

2. [递归神经网络](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=3&sn=c7995dc41eea628204020f4d55c37918&chksm=1bbfe0082cc8691e17064e49c296be4a48652a8914b9644ad4536d8949f1972877566072fcf5&scene=20&xtrack=1#rd)

3. [LSTM、GRU介绍，针对梯度消失、梯度爆炸的解决方案](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=4&sn=f47539af4d10eb067644d46d6827fe5d&chksm=1bbfe0082cc8691e7ebea8e8ad424545b9805dab26d62531bcb7d45171ae75cbaad80c683228&scene=20&xtrack=1#rd)

4. [Memory Network](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=5&sn=e987c36f1726d8db23914fb1160e7828&chksm=1bbfe0082cc8691e2b13b736b8d55a5b41c2a504203e1fb205dfd95ad54168b86ed07822beee&scene=20&xtrack=1#rd)

5. [Text-RNN的原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=6&sn=05d8e60cc79704b843e3448702e4f8ac&chksm=1bbfe0082cc8691eb8b7db5e5ed6dc1cdcc3c5c39b52c7e8bbc95764484fb5520c1528e8fa05&scene=20&xtrack=1#rd)

6. [利用Text-RNN模型来进行文本分类](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=7&sn=42fed67915bc47edf20c9c3ab8db85e2&chksm=1bbfe0082cc8691ecf3317b1c7ffbfb73d3109b4b39d6509a20abf294c692a42d45b6911239f&scene=20&xtrack=1#rd)

7. [Recurrent Convolutional Neural Networks（RCNN）原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001074&idx=8&sn=22583b3592352d0c2b59d280822bc426&chksm=1bbfe0082cc8691e5b54547217e2cdbb55d0b22a578c8261f200d67ac961865cf57a7874c658&scene=20&xtrack=1#rd)

#### 算法代码链接

1.[利用Text-RNN模型来进行文本分类 ](t8RNN/TextRCNN1.py)

### 9、Attention原理    t9Attention

#### 介绍

1. 基本的Attention原理。参考翻译任务中的attention

2. HAN的原理（Hierarchical Attention Networks）

3. 利用Attention模型进行文本分类


#### Requrements

* Python (>=3.5) 

* jieba 

* sklearn

* tensorflow


#### 理论学习

1. [基本的Attention原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001207&idx=2&sn=a978b0afc7a672fd6a4a870c06f7f0f5&chksm=1bbfe08d2cc8699b1f803150faa15b61bb30846b349a239c062343a9e7273aa00fe7c7a19f30&scene=20&xtrack=1#rd)

2. [Attention！注意力机制模型最新综述（附下载）](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001207&idx=3&sn=60afc687f1cb25790d28aad358d76f9c&chksm=1bbfe08d2cc8699b744283c25e7e3535ee3401a54b11ad56c498707662433bb2be806da1668e&scene=20&xtrack=1#rd)

#### 算法代码链接

1.[利用 Bi-LSTM+Attention 模型来进行文本分类 ](t9Attention/Bi-LSTM+Attention/Bi-LSTMAttention.py)


### 10、BERT    t10bert

#### 介绍

1. Transformer的原理

2. BERT的原理

3. 利用预训练的BERT模型将句子转换为句向量，进行文本分类 


#### Requrements

* Python (>=3.5) 

* jieba 

* sklearn

* tensorflow


#### 理论学习

1. [Transformer的原理](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001247&idx=2&sn=db5a96df186f2056c5079f3b3eeba8c0&chksm=1bbfe0e52cc869f30ea5a5f508a79f175ab33cf4acb80fec4699c162c724c8830e14e7fdf337&scene=20&xtrack=1#rd)

2. [放弃幻想，全面拥抱Transformer：NLP三大特征抽取器（CNN/RNN/TF）比较](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001247&idx=3&sn=76f6cb340d5fb16d120fa6643cd2d46c&chksm=1bbfe0e52cc869f33e69114594e053fef38fbc6589d03f6c2b63a3d2fec910786e4b915c31ec&scene=20&xtrack=1#rd)

3. [BERT](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001247&idx=4&sn=17142edb24cf02cc115b8a4cb8d608fa&chksm=1bbfe0e52cc869f3cab5b1bb86bbb0147fe4a81f84fdcbf39b4a4a654b089eeb5f395c4bd8b4&scene=20&xtrack=1#rd)

4. [利用预训练的BERT模型将句子转换为句向量，进行文本分类](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001247&idx=5&sn=cb08789a0487ec4bb623741e113ca420&chksm=1bbfe0e52cc869f386587c221f44eb3837219857bf627d3016e4e6850fc832be1bc60bf4b48d&scene=20&xtrack=1#rd)

#### 算法代码链接

1.[Bert-THUCNews](t10bert/Bert-THUCNews)
