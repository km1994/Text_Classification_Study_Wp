#!/usr/bin/env python
# encoding: utf-8
'''
@author: KM
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: yangkm601@gmail.com
@software: garner
@file: rf_study.py
@time: 2019/4/3 9:44
@desc:  随机森林 分类树和回归树实现
@url:
'''

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np

iris = datasets.load_iris()

#利用随机森林进行分类训练
rfc = RandomForestClassifier(n_estimators=10,max_depth=3)
rfc.fit(iris.data, iris.target)

#利用随机森林进行回归训练
rfr = RandomForestRegressor(n_estimators=10,max_depth=3)
rfr.fit(iris.data, iris.target)

instance = np.array([[4.5,6.7,3.4,5.0]])
print("新样本:", instance)
print('分类结果:', rfc.predict(instance))
print('回归结果:', rfr.predict(instance))