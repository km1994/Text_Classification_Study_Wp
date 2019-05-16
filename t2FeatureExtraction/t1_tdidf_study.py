#encoding=utf-8
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
 
if __name__ == "__main__":
    docs_list=[]
    mywordlist=[]
    stopwords_path = "../resource/stopwords.txt" # 停用词词表

    # 读取文件
    file_object = open('../resource/text1.txt','r',encoding="UTF-8")
    try:
      for line in file_object:
          # 文本分词
          seg_list = jieba.cut(line, cut_all=False)
          liststr="/ ".join(seg_list)

           # 读取停用词文件
          f_stop = open(stopwords_path,'r', encoding='UTF-8')
          try:
            f_stop_text = f_stop.read()
          finally:
            f_stop.close()

          # 停用词清除
          f_stop_seg_list = f_stop_text.split('\n')
          for myword in liststr.split('/'):
            if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
              mywordlist.append(myword)

          docs_list.append(''.join(mywordlist))      # 存入文档列表中
          mywordlist=[]                  # 存入之后，需要清除mywordlist内容，防止重复
    finally:  
      file_object.close()

    print(f"docs_list:{docs_list}")

    docs = np.array(docs_list)
    print(f"docs:{docs}")
    '''
      output:
      docs_list:[
        '当地 时间 2017 15', '日本 神奈川县 横须贺', 
        ' 东芝 国际 反应堆 报废 研究 开发 机构 IRID 共同开发 机器人 公开 亮相', 
        '这个 30 厘米 直径 13 厘米 水下 机器人 投放 福岛 第一 核电站 机组 反应堆 安全壳 底部 展开 调查'
      ]
    docs:[
      '当地 时间 2017 15' '日本 神奈川县 横须贺'
      ' 东芝 国际 反应堆 报废 研究 开发 机构 IRID 共同开发 机器人 公开 亮相'
      '这个 30 厘米 直径 13 厘米 水下 机器人 投放 福岛 第一 核电站 机组 反应堆 安全壳 底部 展开 调查'
    ]
  '''
    # 在scikit-learn中，有两种方法进行TF-IDF的预处理
    # 第一种方法是在用CountVectorizer类向量化之后再调用TfidfTransformer类进行预处理。
    # 第二种方法是直接用TfidfVectorizer完成向量化与TF-IDF预处理。

    # 首先我们来看第一种方法，CountVectorizer+TfidfTransformer的组合，代码如下：
    print("-------------第一种方法-------------")
    count = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count.fit_transform(docs)) 
    print(tfidf)
    '''
        output:
              (0, 16)       0.5
              (0, 20)       0.5
              (0, 2)        0.5
              (0, 1)        0.5
              (1, 19)       0.5773502691896257
              (1, 29)       0.5773502691896257
              (1, 25)       0.5773502691896257
              (2, 5)        0.2982327375202219
              (2, 11)       0.2982327375202219
              (2, 10)       0.2351301157996824
              (2, 18)       0.2982327375202219
              (2, 28)       0.2982327375202219
              (2, 15)       0.2982327375202219
              (2, 22)       0.2982327375202219
              (2, 4)        0.2982327375202219
              (2, 8)        0.2982327375202219
              (2, 21)       0.2351301157996824
              (2, 7)        0.2982327375202219
              (2, 6)        0.2982327375202219
              (3, 10)       0.17972747020412025
              (3, 21)       0.17972747020412025
              (3, 33)       0.22796150660778644
              (3, 3)        0.22796150660778644
              (3, 9)        0.4559230132155729
              (3, 27)       0.22796150660778644
              (3, 0)        0.22796150660778644
              (3, 26)       0.22796150660778644
              (3, 17)       0.22796150660778644
              (3, 30)       0.22796150660778644
              (3, 31)       0.22796150660778644
              (3, 24)       0.22796150660778644
              (3, 23)       0.22796150660778644
              (3, 12)       0.22796150660778644
              (3, 14)       0.22796150660778644
              (3, 13)       0.22796150660778644
              (3, 32)       0.22796150660778644
    '''

    # 第二种方法是直接用TfidfVectorizer完成向量化与TF-IDF预处理
    print("-------------第二种方法-------------")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf2 = TfidfVectorizer()
    re = tfidf2.fit_transform(docs)
    print(re)

    '''
        output:
            -------------第二种方法-------------
              (0, 16)       0.5
              (0, 20)       0.5
              (0, 2)        0.5
              (0, 1)        0.5
              (1, 19)       0.5773502691896257
              (1, 29)       0.5773502691896257
              (1, 25)       0.5773502691896257
              (2, 5)        0.2982327375202219
              (2, 11)       0.2982327375202219
              (2, 10)       0.2351301157996824
              (2, 18)       0.2982327375202219
              (2, 28)       0.2982327375202219
              (2, 15)       0.2982327375202219
              (2, 22)       0.2982327375202219
              (2, 4)        0.2982327375202219
              (2, 8)        0.2982327375202219
              (2, 21)       0.2351301157996824
              (2, 7)        0.2982327375202219
              (2, 6)        0.2982327375202219
              (3, 10)       0.17972747020412025
              (3, 21)       0.17972747020412025
              (3, 33)       0.22796150660778644
              (3, 3)        0.22796150660778644
              (3, 9)        0.4559230132155729
              (3, 27)       0.22796150660778644
              (3, 0)        0.22796150660778644
              (3, 26)       0.22796150660778644
              (3, 17)       0.22796150660778644
              (3, 30)       0.22796150660778644
              (3, 31)       0.22796150660778644
              (3, 24)       0.22796150660778644
              (3, 23)       0.22796150660778644
              (3, 12)       0.22796150660778644
              (3, 14)       0.22796150660778644
              (3, 13)       0.22796150660778644
              (3, 32)       0.22796150660778644

    '''
'''
  参考网址：
    Python读取文件的方法：https://www.jianshu.com/p/d8168034917c
    jieba：https://github.com/fxsjy/jieba
        TF-IDF 创建方法：https://www.cnblogs.com/pinard/p/6693230.html

'''