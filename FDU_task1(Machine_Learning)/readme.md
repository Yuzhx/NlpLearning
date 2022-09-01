# Task1 基于机器学习的文本分类

> 任务要求：使用numpy实现基于logistic/softmax regression的文本分类，其中特征提取部分采取词袋模型和N-Gram模型来分别实现。

-------------

## 特征提取

### 词袋模型

[词袋模型]([词袋模型_百度百科 (baidu.com)](https://baike.baidu.com/item/词袋模型/22776998?fr=aladdin))，实际上考虑了字典之中的每一个字是否出现在句子之中，每个句子可以使用一个长为k的0-1向量matrix来表示，k为字典之中字的个数，matrix[k]=1，表示第k个字出现在句子之中。

实现词袋模型的特征提取，需要首先进行词典的构造，此处的词典实际上是一个列表，列表项为一系列键值对，单词作为键，从0开始的编号作为值，如下图所示为部分字典：

<img alt="image-20220808232534923" src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202208082325952.png" style="width: 50%;"/>

构造字典的代码如下：

```python
def get_dict(self):  # 得到字典
	for term in self.data:  # 遍历数据集
		sentence = term[2]
		sentence = sentence.upper()  # 将句子转化为大写
		words = sentence.split()  # 将句子分为单词组
		for word in words:
			if word not in self.dict_words:  # 对于当前不存在句子之中的单词
				self.dict_words[word] = len(self.dict_words) # 加入键值对之中
	self.len = len(self.dict_words) #字典长度，即句子向量长
	self.test_matrix = numpy.zeros((len(self.test), self.len))  # 训练集矩阵初始化
	self.train_matrix = numpy.zeros((len(self.train), self.len))  # 测试集矩阵初始化
```

构造完字典之后，需要将数据集之中的句子转化为特征向量的形式，即检查句子中包含有哪些单词，并且将单词对应的再特征向量的那一位置为1：

```python
def get_matrix(self):
	for i in range(len(self.train)):  # 训练集矩阵
		s = self.train[i][2]
		words = s.split()
		for word in words:
			word = word.upper()
			self.train_matrix[i][self.dict_words[word]] = 1
	for i in range(len(self.test)):  # 测试集矩阵
        s = self.test[i][2]
        words = s.split()
        for word in words:
            word = word.upper()
            self.test_matrix[i][self.dict_words[word]] = 1
```

最终得到的特征向量如下图所示：

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202208091129125.png" alt="image-20220809112930075" style="width: 50%;" />

每行代表一个句子。

### N-Gram模型

[N-Gram]([N-Gram_百度百科 (baidu.com)](https://baike.baidu.com/item/N-Gram/10803752?fr=aladdin))在词袋模型的基础之上利用上下文中相邻词间的搭配信息，将相邻N个词组成的词组也加入到字典之中，例如"Fly me to the moon"的2元N-Gram模型的字典为："FLY, ME, TO, THE, MOON, FLY_ME, ME_TO, TO_THE, THE_MOON"，所以构造字典部分需要进行修改：

```python
def get_dict(self):
	for d in range(1, self.dimension + 1):  # 提取 1-gram, 2-gram,..., N-gram 特征
		print("1")
		for term in self.data:
			s = term[2]
			s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
			words = s.split()
			for i in range(len(words) - d + 1):  # 一个一个特征找
				temp = words[i:i + d]
				temp = "_".join(temp)  # 形成i d-gram 特征
				if temp not in self.dict_words:
					self.dict_words[temp] = len(self.dict_words)
	self.len = len(self.dict_words)
	self.test_matrix = numpy.zeros((len(self.test), self.len))  # 训练集矩阵初始化
	self.train_matrix = numpy.zeros((len(self.train), self.len))  # 测试集矩阵初始化
```

其中第10行的[join()]([Python join()方法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python/att-string-join.html))方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

同样地，在生成特征矩阵时，同样遍历句子的所有单词，双单词词组...N单词词组，并且将其同样使用join方法，将_作为连接符，并且在字典之中查询单词和词组是否存在，将特征矩阵之中对应的标志位置为1

```python
def get_matrix(self):
    for d in range(1, self.dimension + 1):
        for i in range(len(self.train)):  # 训练集矩阵
            s = self.train[i][2]
            s = s.upper()
            words = s.split()
            for j in range(len(words) - d + 1):
                temp = words[j:j + d]
                temp = "_".join(temp)
                self.train_matrix[i][self.dict_words[temp]] = 1
        for i in range(len(self.test)):  # 测试集矩阵
            s = self.test[i][2]
            s = s.upper()
            words = s.split()
            for j in range(len(words) - d + 1):
                temp = words[j:j + d]
                temp = "_".join(temp)
                self.test_matrix[i][self.dict_words[temp]] = 1
```

最终得到m行n列的矩阵，m为训练集/测试集的大小，n为字典的大小，即self.train_matrix与self.test_matrix

## softmax 逻辑回归

[softmax逻辑回归](https://baike.baidu.com/item/softmax%20%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/22689563?fr=aladdin)模型是logistic回归模型在多分类问题上的推广，在多分类问题中，类标签y可以取两个以上的值。

在回归计算过程之中，所送入分类器的是train_matrix或test_matrix，其行数为训练集/测试集的大小，列数为字典大小，包括预测结果以及梯度下降两个工作。

当进行结果预测时，以[707, 363]规模的训练集为例，分类器中的核心是一个规模为[363, 5]的参数矩阵W，对于训练集之中的一个句子X[k]（规模为[1, 363]），将X[k]和W相乘，得到[1, 5]的yhat，便为softmax的结果

当进行梯度下降时，实际上执行的是：
$$
W=W+\alpha*\frac{\sum^{m}_{i=1}{X_i^T(y-yhat)} }{m}
$$
其中m为一次训练之中的数据总量，其代码如下所示：

```python
yhat = self.softmax_calculation(X[k].reshape(1,-1).dot(self.W))
increment += X[k].reshape(-1, 1).dot(self.change_y(y[k]) - yhat)  # 梯度加和
```

其中increment是这一次训练过程中得到的梯度，多次训练的梯度取平均，乘上系数α之后与W相加，得到新的W



