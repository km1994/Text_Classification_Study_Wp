# encoding=utf-8
'''

    功能：
    参考：
        [python] LDA处理文档主题分布代码入门笔记：https://blog.csdn.net/eastmount/article/details/50824215

'''

# 1.载入数据
import numpy as np
import lda
import lda.datasets

# document-term matrix
X = lda.datasets.load_reuters()
# print("type(X): {}".format(type(X)))
# print("shape: {}\n".format(X.shape))
# print(X[:5, :5])

# the vocab
vocab = lda.datasets.load_reuters_vocab()
# print("type(vocab): {}".format(type(vocab)))
# print("len(vocab): {}\n".format(len(vocab)))
# print(vocab[:5])

# titles for each story
titles = lda.datasets.load_reuters_titles()
# print("type(titles): {}".format(type(titles)))
# print("len(titles): {}\n".format(len(titles)))
'''
    output:
        type(X): <class 'numpy.ndarray'>
        shape: (395, 4258)

        [[ 1  0  1  0  0]
         [ 7  0  2  0  0]
         [ 0  0  0  1 10]
         [ 6  0  1  0  0]
         [ 0  0  0  2 14]]
        type(vocab): <class 'tuple'>
        len(vocab): 4258

        ('church', 'pope', 'years', 'people', 'mother')
        type(titles): <class 'tuple'>
        len(titles): 395
'''

# X[0,3117] is the number of times that word 3117 occurs in document 0
doc_id = 0
word_id = 3117
# print("doc id: {} word id: {}".format(doc_id, word_id))
# print("-- count: {}".format(X[doc_id, word_id]))
# print("-- word : {}".format(vocab[word_id]))
# print("-- doc  : {}".format(titles[doc_id]))
'''
    output:
        doc id: 0 word id: 3117
        -- count: 2
        -- word : heir-to-the-throne
        -- doc  : 0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20
'''
# 2.训练模型
# 其中设置20个主题，500次迭代
model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available

# 3.主题-单词（topic-word）分布
# 代码如下所示，计算'church', 'pope', 'years'这三个单词在各个主题(n_topocs=20，
# 共20个主题)中的比重，同时输出前5个主题的比重和，其值均为1。
topic_word = model.topic_word_
# print("type(topic_word): {}".format(type(topic_word)))
# print("shape: {}".format(topic_word.shape))
# print(vocab[:3])
# print(topic_word[:, :3])
#
# for n in range(5):
#     sum_pr = sum(topic_word[n, :])
#     print("topic: {} sum: {}".format(n, sum_pr))
'''
    output:
        type(topic_word): <class 'numpy.ndarray'>
        shape: (20, 4258)
        ('church', 'pope', 'years')
        [[2.72436509e-06 2.72436509e-06 2.72708945e-03]
         [2.29518860e-02 1.08771556e-06 7.83263973e-03]
         ...
         [2.39373034e-06 2.39373034e-06 2.39373034e-06]
         [3.32493234e-06 3.32493234e-06 3.32493234e-06]]
        topic: 0 sum: 1.0000000000000875
        topic: 1 sum: 1.0000000000001148
        topic: 2 sum: 0.9999999999998656
        topic: 3 sum: 1.0000000000000042
        topic: 4 sum: 1.0000000000000928
'''

# 4.计算各主题Top-N个单词
# 下面这部分代码是计算每个主题中的前5个单词
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n + 1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

# 5.文档-主题（Document-Topic）分布
# 计算输入前10篇文章最可能的Topic
doc_topic = model.doc_topic_
# print("type(doc_topic): {}".format(type(doc_topic)))
# print("shape: {}".format(doc_topic.shape))
# for n in range(10):
#     topic_most_pr = doc_topic[n].argmax()
#     print("doc: {} topic: {}".format(n, topic_most_pr))

'''
    output:
        *Topic 0
        - government british minister west group
        *Topic 1
        - church first during people political
        *Topic 2
        - elvis king wright fans presley
        *Topic 3
        - yeltsin russian russia president kremlin
        *Topic 4
        - pope vatican paul surgery pontiff
        *Topic 5
        - family police miami versace cunanan
        *Topic 6
        - south simpson born york white
        *Topic 7
        - order church mother successor since
        *Topic 8
        - charles prince diana royal queen
        *Topic 9
        - film france french against actor
        *Topic 10
        - germany german war nazi christian
        *Topic 11
        - east prize peace timor quebec
        *Topic 12
        - n't told life people church
        *Topic 13
        - years world time year last
        *Topic 14
        - mother teresa heart charity calcutta
        *Topic 15
        - city salonika exhibition buddhist byzantine
        *Topic 16
        - music first people tour including
        *Topic 17
        - church catholic bernardin cardinal bishop
        *Topic 18
        - harriman clinton u.s churchill paris
        *Topic 19
        - century art million museum city
        type(doc_topic): <class 'numpy.ndarray'>
        shape: (395, 20)
        doc: 0 topic: 8
        doc: 1 topic: 1
        doc: 2 topic: 14
        doc: 3 topic: 8
        doc: 4 topic: 14
        doc: 5 topic: 14
        doc: 6 topic: 14
        doc: 7 topic: 14
        doc: 8 topic: 14
        doc: 9 topic: 8
'''

# 6.两种作图分析
# 详见英文原文，包括计算各个主题中单词权重分布的情况：
import matplotlib.pyplot as plt

f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0, 5, 9, 14, 19]):
    ax[i].stem(topic_word[k, :], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-50, 4350)
    ax[i].set_ylim(0, 0.08)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()

# 第二种作图是计算文档具体分布在那个主题，代码如下所示：
import matplotlib.pyplot as plt

f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([1, 3, 4, 8, 9]):
    ax[i].stem(doc_topic[k, :], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 21)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()
