from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import sys
from sklearn.decomposition import LatentDirichletAllocation

'''
    工具包 begin
'''
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

'''
    工具包 end
'''
'''
    文件读写 begin
'''
#读取文件数据
def read_file(filename,demo_flag = False):
    '''
    读取文件数据
    :param filename:    String 文件名称包含路径
    :param demo_flag:   String True 只读取 1000 样本数据，Fasle 读取全部数据
    :return:
        contents:   list    内容列表
        labels:     list    内容标签
    '''
    contents, labels = [], []
    contents_num = 0
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(''.join(list(native_content(content))))
                    labels.append(native_content(label))
                contents_num = contents_num + 1
                if demo_flag and contents_num == 1000:
                    break
            except:
                pass
    return contents, labels

# 获取训练数据，测试数据
train_data,labels = read_file(filename='../resource/THUCNews_ch/t1_cut_words_cnews.train_demo.txt',demo_flag = False)
train_data = np.array(train_data)
#
# test_data = []
# with open('./after_preprocess_testdata.txt',encoding='utf-8') as f:
#     for line in f:
#         test_data.append(line.strip())

# 获取标签编码
df1 = pd.read_csv('../resource/THUCNews_ch/t1_cut_words_cnews.train_demo.txt',sep='\t',names=['label','content'],encoding='UTF-8',engine='python')
df2 = pd.read_csv('../resource/THUCNews_ch/t1_cut_words_cnews.train_demo.txt',sep='\t',names=['label','content'],encoding='UTF-8',engine='python')
encoder = LabelEncoder()


train_y = encoder.fit_transform(df1['label'])
test_y = encoder.transform(df2['label'])



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bow = cv.fit_transform(train_data)
print(bow)

# 这里是根据词袋，计算与标签的互信息，取相关性最大的前5000个词
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(mutual_info_classif, k=5000)
new_train_x = selector.fit_transform(bow,train_y)
print(new_train_x.shape)

# 保存剩余筛选器模型
pkl = pickle.dumps(selector)
with open('./selector_model.pkl','wb') as f:
    f.write(pkl)

# 测试读取筛选器
with open('./selector_model.pkl','rb') as f:
    model = pickle.loads(f.read())
new_train_x = model.transform(bow)
print(new_train_x.shape)

#开始创建LDA模型
lda = LatentDirichletAllocation(n_components=100,max_iter=800,random_state=1)
lda.fit(new_train_x)
def load_lda(path):
    with open(path,'rb') as f:
        lda = pickle.loads(f.read())
    return lda

lda = load_lda('./lda_model.pkl')

lda_feature = lda.transform(new_train_x)
print(lda_feature[0])


print(lda_feature.shape)

train_X = np.concatenate((new_train_x.toarray(),lda_feature),axis=1)
print(train_X.shape)


from sklearn.naive_bayes import GaussianNB
gs = GaussianNB()
gs.fit(train_X,train_y)
print(gs.score(train_X,train_y))


test_x_bag = cv.transform(test_data)
test_x_bag = model.transform(test_x_bag)
test_x_lda = lda.transform(test_x_bag)
test_X = np.concatenate((test_x_bag.toarray(),test_x_lda),axis=1)
print(test_X.shape)


pred_y = gs.predict(test_X)
acc = np.sum(pred_y == test_y)/len(test_y)
print('acc', acc)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(test_y,pred_y))
print(confusion_matrix(test_y,pred_y))


# 不拼接lda特征
train_X = new_train_x.toarray()
gs = GaussianNB()
gs.fit(train_X,train_y)
test_x_bag = cv.transform(test_data)
test_x_bag = model.transform(test_x_bag)
test_X = test_x_bag.toarray()
print(test_X.shape)
print(gs.score(train_X,train_y))
print(gs.score(test_X,test_y))
print()

#  采用tfidf+lda+朴素贝叶斯
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf_train_x = transformer.fit_transform(new_train_x)

tfidf_train_X = np.concatenate((tfidf_train_x.toarray(),lda_feature),axis=1)
# tfidf_train_X = tfidf_train_x.toarray()

gs2 = GaussianNB()
gs2.fit(tfidf_train_X,train_y)



pred = gs2.predict(tfidf_train_X)
print(gs2.score(tfidf_train_X,train_y))
report = classification_report(train_y,pred)
print(report)


test_x_tfidf = transformer.fit_transform(test_x_bag)
test_x_tfidf_lda = lda.transform(test_x_tfidf)
test_tfidf_X = np.concatenate((test_x_tfidf.toarray(),test_x_tfidf_lda),axis=1)
# test_tfidf_X = test_x_tfidf.toarray()
print(gs2.score(test_tfidf_X,test_y))
pred = gs2.predict(test_tfidf_X)
report = classification_report(test_y,pred)
print(report)