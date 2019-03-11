### 任务四 

1. 朴素贝叶斯的原理
2. 利用朴素贝叶斯模型进行文本分类
3. SVM的原理
4. 利用SVM模型进行文本分类
5. pLSA、共轭先验分布；LDA主题模型原理
6. 使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类
7. 参考
  朴素贝叶斯1：sklearn：https://blog.csdn.net/u013710265/article/details/72780520
  lda2：https://blog.csdn.net/u013710265/article/details/73480332，http://www.cnblogs.com/pinard/p/6908150.html，https://blog.csdn.net/chen_yiwei/article/details/88370526
  SVM原理篇之手撕SVM(里面有手写SVM代码，可自行去实现一次)：https://blog.csdn.net/weixin_39605679/article/details/81170300



| 文件名                | 内容                                                        |
| --------------------- | ----------------------------------------------------------- |
| preprocess_data.ipynb | cnews数据预处理代码                                         |
| naive_bayes.ipynb     | 朴素贝叶斯对cnews分类(直接使用了tfidf，未添加lda生成的特征) |
| add_lda.ipynb         | 训练lda模型，生成特征，再合并特征，进行分类                 |
| svm_test.ipynb        | svm测试代码                                                 |
|                       |                                                             |



