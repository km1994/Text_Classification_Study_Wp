from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle

# 获取训练数据，测试数据
train_data = []
with open('./after_preprocess_traindata.txt',encoding='utf-8') as f:
    for line in f:
        train_data.append(line.strip())

test_data = []
with open('./after_preprocess_testdata.txt',encoding='utf-8') as f:
    for line in f:
        test_data.append(line.strip())

# 获取标签编码
df1 = pd.read_csv('../dataset/cnews/cnews.train.txt',sep='\t',names=['label','content'],encoding='UTF-8',engine='python')
df2 = pd.read_csv('../dataset/cnews/cnews.test.txt',sep='\t',names=['label','content'],encoding='UTF-8',engine='python')
encoder = LabelEncoder()

train_y = encoder.fit_transform(df1['label'])
test_y = encoder.transform(df2['label'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bow = cv.fit_transform(train_data)

with open('./selector_model.pkl','rb') as f:
    model = pickle.loads(f.read())
new_train_x = model.transform(bow)
print(new_train_x.shape)

print('start lda')

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=100,max_iter=10,random_state=1)
lda.fit(new_train_x)

print('after lda')

with open('./lda_model.pkl','wb') as f:
    lda_pkl = pickle.dumps(lda)
    f.write(lda_pkl)
