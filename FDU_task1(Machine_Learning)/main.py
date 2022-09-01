import numpy
import random
import csv
# 特征提取部分
def data_split(data, test_rate=0.3, max_item=1000):
    """把数据按一定比例划分成训练集和测试集"""
    train = list()
    test = list()
    i = 0
    for datum in data:
        i += 1
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
        if i > max_item:
            break
    return train, test


class Bag:
    """Bag of words"""
    def __init__(self, my_data, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item=max_item
        self.dict_words = dict()  # 单词到单词编号的映射
        self.len = 0  # 记录有几个单词
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = None  # 训练集的0-1矩阵（每行一个句子，每个句子表示为长度为k的向量）
        self.test_matrix = None  # 测试集的0-1矩阵（每行一个句子，每个句子表示为长度为k的向量）

    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
            words = s.split()
            for word in words:  # 一个一个单词寻找
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = numpy.zeros((len(self.test), self.len))  # 初始化0-1矩阵
        self.train_matrix = numpy.zeros((len(self.train), self.len))  # 初始化0-1矩阵

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


class Gram:
    """N-gram"""
    def __init__(self, my_data, dimension=2, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()  # 特征到t正编号的映射
        self.len = 0  # 记录有多少个特征
        self.dimension = dimension  # 决定使用几元特征
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = None  # 训练集0-1矩阵（每行代表一句话）
        self.test_matrix = None  # 测试集0-1矩阵（每行代表一句话）

    def get_words(self):
        for d in range(1, self.dimension + 1):  # 提取 1-gram, 2-gram,..., N-gram 特征
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

    def get_matrix(self):
        for d in range(1, self.dimension + 1):
            for i in range(len(self.train)):  # 训练集矩阵
                s = self.train[i][2]
                s = s.upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    a=self.train_matrix[i]
                    self.train_matrix[i][self.dict_words[temp]] = 1
            for i in range(len(self.test)):  # 测试集矩阵
                s = self.test[i][2]
                s = s.upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.test_matrix[i][self.dict_words[temp]] = 1

class Softmax:
    """Softmax regression"""
    def __init__(self, sample, typenum, feature):
        self.sample = sample  # 训练集样本个数
        self.typenum = typenum  # （情感）种类个数
        self.feature = feature  # 0-1向量的长度
        self.W = numpy.random.randn(feature, typenum)  # 参数矩阵W初始化

    def softmax_calculation(self, x):
        """x是向量，计算softmax值"""
        exp = numpy.exp(x - numpy.max(x))  # 先减去最大值防止指数太大溢出
        return exp / exp.sum()

    def softmax_all(self, wtx):
        """wtx是矩阵，即许多向量叠在一起，按行计算softmax值"""
        wtx -= numpy.max(wtx, keepdims=True) # 先减去行最大值防止指数太大溢出
        wtx = numpy.exp(wtx)
        wtx /= numpy.sum(wtx, keepdims=True)
        return wtx

    def change_y(self, y):
        """把（情感）种类转换为一个one-hot向量"""
        ans = numpy.array([0] * self.typenum)
        ans[y] = 1
        return ans.reshape(1,-1)

    def prediction(self, X):
        """给定0-1矩阵X，计算每个句子的y_hat值（概率）"""
        prob = self.softmax_all(X.dot(self.W))
        return prob.argmax()

    def correct_rate(self, train, train_y, test, test_y):
        """计算训练集和测试集的准确率"""
        # train set
        n_train = len(train)
        pred_train = self.prediction(train)
        train_correct = sum([train_y[i] == pred_train[i] for i in range(n_train)]) / n_train
        # test set
        n_test = len(test)
        pred_test = self.prediction(test)
        test_correct = sum([test_y[i] == pred_test[i] for i in range(n_test)]) / n_test
        print(train_correct, test_correct)
        return train_correct, test_correct

    def regression(self, X, y, alpha, times, strategy="mini", m=100):
        """Softmax regression"""
        if self.sample != len(X) or self.sample != len(y):
            raise Exception("Sample size does not match!")  # 样本个数不匹配
        if strategy == "mini":  # mini-batch
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  # 梯度初始为0矩阵
                for j in range(m):  # 随机抽m次
                    k = random.randint(0, self.sample - 1)
                    yhat = self.softmax_calculation(X[k].reshape(1,-1).dot(self.W))
                    increment += X[k].reshape(-1, 1).dot(self.change_y(y[k]) - yhat)  # 梯度加和
                # print(i * m)
                self.W += alpha / m * increment  # 参数更新
        elif strategy == "shuffle":  # 随机梯度
            for i in range(times):
                k = random.randint(0, self.sample - 1)  # 每次抽一个
                yhat = self.softmax_calculation(X[k].reshape(1,-1).dot(self.W))
                increment = X[k].reshape(-1, 1).dot(self.change_y(y[k]) - yhat)  # 计算梯度
                self.W += alpha * increment  # 参数更新
                # if not (i % 10000):
                #     print(i)
        elif strategy=="batch":  # 整批量梯度
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  ## 梯度初始为0矩阵
                for j in range(self.sample):  # 所有样本都要计算
                    yhat = self.softmax_calculation(X[j].reshape(1,-1).dot(self.W))
                    increment += X[j].reshape(-1, 1).dot(self.change_y(y[j]) - yhat)  # 梯度加和
                # print(i)
                self.W += alpha / self.sample * increment  # 参数更新
        else:
            raise Exception("Unknown strategy")

def gradient_plot(bag,gram, total_times, mini_size, alpha):

    # Bag of words

    # Shuffle
    shuffle_train = list()
    shuffle_test = list()
    soft = Softmax(len(bag.train), 5, bag.len)
    soft.regression(bag.train_matrix, bag.train_y, alpha, total_times, "shuffle")
    r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
    shuffle_train.append(r_train)
    shuffle_test.append(r_test)

    # Batch
    batch_train = list()
    batch_test = list()
    soft = Softmax(len(bag.train), 5, bag.len)
    soft.regression(bag.train_matrix, bag.train_y, alpha, int(total_times/bag.max_item), "batch")
    r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
    batch_train.append(r_train)
    batch_test.append(r_test)

    # Mini-batch
    mini_train = list()
    mini_test = list()
    soft = Softmax(len(bag.train), 5, bag.len)
    soft.regression(bag.train_matrix, bag.train_y, alpha, int(total_times/mini_size), "mini",mini_size)
    r_train, r_test= soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
    mini_train.append(r_train)
    mini_test.append(r_test)

    # N-gram
    # Shuffle
    shuffle_train = list()
    shuffle_test = list()
    soft = Softmax(len(gram.train), 5, gram.len)
    soft.regression(gram.train_matrix, gram.train_y, alpha, total_times, "shuffle")
    r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
    shuffle_train.append(r_train)
    shuffle_test.append(r_test)

    # Batch
    batch_train = list()
    batch_test = list()
    soft = Softmax(len(gram.train), 5, gram.len)
    soft.regression(gram.train_matrix, gram.train_y, alpha, int(total_times / gram.max_item), "batch")
    r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
    batch_train.append(r_train)
    batch_test.append(r_test)

    # Mini-batch
    mini_train = list()
    mini_test = list()
    soft = Softmax(len(gram.train), 5, gram.len)
    soft.regression(gram.train_matrix, gram.train_y, alpha, int(total_times / mini_size), "mini", mini_size)
    r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
    mini_train.append(r_train)
    mini_test.append(r_test)

# 数据读取
with open('train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

# 初始化
data = temp[1:]
max_item=1000
random.seed(2021)
numpy.random.seed(2021)

# 特征提取
bag=Bag(data,max_item)
bag.get_words()
bag.get_matrix()

gram=Gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()

gradient_plot(bag,gram,10000,10)  # 计算10000次
gradient_plot(bag,gram,100000,10)  # 计算100000次