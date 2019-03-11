Task3 特征选择  
1. TF-IDF原理。
2. 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）
3. 互信息的原理。
4. 使用第二步生成的特征矩阵，利用互信息进行特征筛选。
5. 参考
    文本挖掘预处理之TF-IDF：https://www.cnblogs.com/pinard/p/6693230.html
    使用不同的方法计算TF-IDF值：https://www.jianshu.com/p/f3b92124cd2b
    sklearn-点互信息和互信息：https://blog.csdn.net/u013710265/article/details/72848755
    如何进行特征选择（理论篇）机器学习你会遇到的“坑”：https://baijiahao.baidu.com/s?id=1604074325918456186&wfr=spider&for=pc



### 1、TF-IDF原理

假设我们有这么4个短文本：

```python
corpus=[
    "I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"
] 
```



#### 词袋模型

一句话的语义很大程度取决于某个单词出现的次数，所以可以把句子中所有可能出现的单词作为特征名，每一个句子为一个样本，单词在句子中出现的次数为特征值构建数学模型。该模型即称为**词袋模型**。

​	先统计处所有出现过的词(未去除停用词)

```python
['a','and','apple','car','china','china','come','i','in','is','love','papers','polupar','science','some','tea','the','this','to','travel','work','write']
```

```python
# 统计出词频
[[0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
 [1 0 0 1 1 0 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
 [0 1 1 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
```

#### 词频（TF）

​	单词在句子中出现的次数除以句子的总词数称为词频。即一个单词在一个句子中出现的频率。词频相比单词出现的次数更可以客观的评估单词对一句话语义的贡献度。对词袋矩阵归一化处理即可得到单词的词频。

$$TF_w=\dfrac{某一类中词条w出现的次数}{该类中的所有词条数目}$$

​	但是，像是to这种词在一句话里经常出现多次，然而实际上”to“是一个非常普遍的词，几乎所有的文本都会用到，因此虽然它的词频为2，但是重要性却比词频为1的"China"和“Travel”要低的多。如果仅仅凭借词频是无法反映这一点的，所以我们可以对此进行TF-IDF。



#### TF-IDF介绍：

​	<font color=red><b>TF-IDF= TF*DF</b></font>

​	TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率) 是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

​	文档频率(DF)：$DF=\dfrac{含有某个单词的文档样本数量}{总文档样本数量}$    文档频率越低，代表某个单词对语义的贡献越高。

​	逆文档频率（IDF） $IDF=log(\dfrac{总文档样本数量}{含有某个单词的文档样本数量 + 1})$ 

​	逆向文件频率 (inverse document frequency, IDF) IDF的主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。

​	<font color=red>某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。</font>



TF-IDF生成的方法：

```python
#方法一：先生成词袋bow，再通过tfidf转化器转换
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"
]
# 词袋模型对象
cv = CountVectorizer()
# 生成词袋,并转换成数组形式
bow = cv.fit_transform(corpus)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(bow)
print(tfidf)
#  (0, 16)	0.4424621378947393
#  (0, 15)	0.697684463383976
#  (0, 4)	0.4424621378947393
```

```python
# 方法二：直接创建TfidfVectorizer类
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    "I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"
]
tfidf = TfidfVectorizer()
data = tfidf.fit_transform(corpus) #会自动删除单个字符词如：a , i 
print(data)
# 默认稀疏矩阵模式
# (0, 4)	0.4424621378947393
# (0, 15)	0.697684463383976
# (0, 3)	0.348842231691988
# (0, 16)	0.4424621378947393
# (1, 3)	0.3574550433419527
# (1, 14)	0.45338639737285463

print(tfidf.get_feature_names())
# ['and','apple','car','china','come','in','is','love','papers','polupar','science',
# 'some','tea','the','this','to','travel','work','write']

```

Hash Trick小知识：http://www.cnblogs.com/pinard/p/6688348.html



### 点互信息(PMI) 和 互信息(MI)

1、机器学习相关文献里面，经常会用到点互信息PMI(Pointwise Mutual Information)这个指标来衡量两个事物之间的相关性（比如两个词）。

​					$$PMI(x;y) = log(\dfrac{p(x,y)}{p(x)p(y)}) = log(\dfrac{p(x|y)}{p(x)}) = log(\dfrac{p(y|x)}{p(y)})$$

p(x)：单词x在文档中出现的概率。 p(y)：单词y在文档中出现的概率。 p(x,y)：单词x,y同时出现在一个文档的概率

由公式可知:

​	当x,y相互独立时，PMI = log1 = 0

​	当x,y相关性越大，p(x,y)就相比于p(x)p(y)越大，即PMI值越大。



2、点互信息PMI其实就是从信息论里面的互信息这个概念里面衍生出来的。  

​	互信息：

​					$$I(X;Y)=\sum_x\sum_yp(x,y)log(\dfrac{p(x,y)}{p(x)p(y)})$$

​	**互信息衡量的是两个随机变量之间的相关性，即一个随机变量中包含的关于另一个随机变量的信息量。**

​	**可以看出互信息其实就是对X和Y的所有可能的取值情况的点互信息PMI的加权和。**



3、python sklearn工具包求互信息：

```python
from sklearn import metrics as mr
mr.mutual_info_score(label,x)
```

 

### 特征选择

一个典型的机器学习任务，是通过样本的特征来预测样本所对应的值。如果样本的特征少了，我们会考虑增加特征，比如多项式回归(Polynomial Regression)就是典型的增加特征的算法。但是，当特征过多时，就会导致模型的复杂度也就越高，越容易导致过拟合。

现实中的情况，往往是特征太多了，需要减少一些特征。在此我们主要减少的两类特征：

​	**1、无关特征**： 特征与目标结果的状态完全无关

​		通过空气的湿度，环境的温度，风力和当地人的男女比例来预测明天会不会下雨，其中男女比例就是典型的无关特征。

​	**2、多余特征**： 两特征之间存在强相关性

​		通过房屋的面积，卧室的面积，车库的面积，所在城市的消费水平，所在城市的税收水平等特征来预测房价，那么消费水平（或税收水平）就是多余特征。因为消费水平和税收水平存在相关性，我们只需要其中一个特征就够了。



减少特征的意义：

​	1、降低过拟合

​	2、特征选择可以使模型获得更好的解释性

​	3、加快模型的训练速度

​	4、得到更好的性能

特征选择常见的方法：

​	**1、过滤法**

​		过滤法只用于检验**特征向量**和**目标（响应变量）**的相关度，不需要任何的机器学习的算法，不依赖于任何模型，只是应用统计量做筛选：我们根据统计量的大小，设置合适的阈值，将低于阈值的特征剔除。

​		**协方差**  $$Cov(x,y)=E[(X - EX)(Y - EY)] = E(XY)-E(X)E(Y)$$ 

​		如果X和Y相互独立，则有 : $$E(XY)=E(X)E(Y),	 Cov(X,Y)=0$$ (反之不成立)

​		**相关系数** $$\rho = \dfrac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$$  取值去间 [-1,1]， $\rho=0$表示不相关

​		当特征向量和目标（响应变量）的相关度 $\rho$越接近于0，则该特征越无用。

​	**2、包裹法**

|   方法   |                             描述                             |
| :------: | :----------------------------------------------------------: |
| 前向搜索 | 在开始时，按照特征数来划分子集，每个子集只有一个特征，对每个子集进行评价。然后在最优的子集上逐步增加特征，使模型性能提升最大，直到增加特征并不能使模型性能提升为止。 |
| 后向搜索 | 在开始时，将特征集合分别减去一个特征作为子集，每个子集有N—1个特征，对每个子集进行评价。然后在最优的子集上逐步减少特征，使得模型性能提升最大，直到减少特征并不能使模型性能提升为止。 |
| 双向搜索 |         将Forward search 和Backward search结合起来。         |
| 递归剔除 | 反复的训练模型，并剔除每次的最优或者最差的特征，将剔除完毕的特征集进入下一轮训练，直到所有的特征被剔除，被剔除的顺序度量了特征的重要程度。 |

​			在开始时，按照特征数来划分子集，每个子集只有一个特征，对每个子集进行评价。然后在最优的子集上逐步增加特征，使模型性能提升最大，直到增加特征并不能使模型性能提升为止。

​		后向搜索

​	**3、嵌入法**   

​		1、加入正则项 L1,L2。通过降低权重系数的办法来降低过拟合，那么在线性模型中，降低权重系数就意味着与之相关的特征并不重要，实际上就是对特征做了一定的筛选。

​		2、决策树也是典型的嵌入法。因为决策树是利用一个特征进行分类，我们在生成决策树的过程就是挑选特征的过程，并且根据特征的不同取值构建子节点，直到特征没有分类能力或者很小，就停止生成节点。



更多参考资料：https://www.cnblogs.com/stevenlk/p/6543628.html