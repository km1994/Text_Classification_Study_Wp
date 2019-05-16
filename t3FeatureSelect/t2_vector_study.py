
#encoding=utf-8
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
 
if __name__ == "__main__":
    count = CountVectorizer()

    docs_list=[]
    mywordlist=[]
    stopwords_path = "../resource/stopwords.txt" # 停用词词表

    # 读取文件
    file_object = open('../resource/text1.txt','r', encoding='UTF-8')
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

    #创建词袋模型的词汇库
    bag = count.fit_transform(docs)
    #查看词汇的位置，词汇是以字典的形式存储
    print(count.vocabulary_)
    '''
      output:
        {'当地': 16, '时间': 20, '2017': 2, '15': 1, '日本': 19, 
        '神奈川县': 29, '横须贺': 25, '东芝': 5, '国际': 11, 
        '反应堆': 10, '报废': 18, '研究': 28, '开发': 15, '机构': 22, 
        'irid': 4, '共同开发': 8, '机器人': 21, '公开': 7, '亮相': 6, 
        '这个': 33, '30': 3, '厘米': 9, '直径': 27, '13': 0, '水下': 26, 
        '投放': 17, '福岛': 30, '第一': 31, '核电站': 24, '机组': 23, 
        '安全壳': 12, '底部': 14, '展开': 13, '调查': 32}
    '''
    print(bag.toarray())
    '''
      output:
        [[0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0]
       [0 0 0 0 1 1 1 1 1 0 1 1 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0]
       [1 0 0 1 0 0 0 0 0 2 1 0 1 1 1 0 0 1 0 0 0 1 0 1 1 0 1 1 0 0 1 1 1 1]]
    '''