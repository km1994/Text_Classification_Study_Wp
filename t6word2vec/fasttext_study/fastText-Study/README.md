# fastText-Study
{https://github.com/ruanyangry/fastText-Study}

    学习fastText进行文本分类，对wiki中文数据集进行分类操作。  

    wikizh.py: 对https://dumps.wikimedia.org/zhwiki/20180801/ 上下载的语料进行预处理操作  
               参考：https://kexue.fm/archives/4176  
           
    wikizhdeal.py: 对wikizh.py 输出的 wiki.txt 进行处理，生成 train,test,validation 数据集，本例演示的是二分类问题  

    train_supervised_ruanyang.py: 利用fastText中的 train_supervised 进行模型训练，最后对输入文本进行预测操作  

    fasttext_linux_py: 基于python中的subprocess模块调用单独编译的 linux 版本 fasttext，等价于官网上的基于命令行的调用形式  

    wikizh_cnn_classifier_v1.py: 基于 word 级别的預训练词向量，使用CNN算法进行分类模型训练  

    wikizh_cnn_classifier_v2.py: 利用 jieba 分词，使用ngram的預训练词向量，使用CNN算法进行分类模型训练  

# 預安装软件  

    https://github.com/facebookresearch/fastText;  
    https://github.com/pybind/pybind11;  
    https://github.com/RaRe-Technologies/gensim;  
    https://github.com/fxsjy/jieba;  

# 目的  

    1. 熟悉fastText 进行文本分类的操作  
    2. text CNN 算法基于词和分词实现  

# 預训练词向量下载  

    https://github.com/Embedding/Chinese-Word-Vectors  


