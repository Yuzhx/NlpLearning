### pytorch神经网络的构建

任务：进行一元四次函数的拟合

使用module来定义神经网络：

```python
import torch
import torch.nn.functional as f
class Net(torch.nn.Module):
    def __init__(self):
        self.hidden1 = nn.Linear(1, 40)
        self.hidden2 = nn.Linear(40, 10)
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = f.relu(self.hidden1(x))
        x = f.relu(self.hidden2(x))
        output = self.regression(x)
        return output
```

其中需要定义`__init()__`和`forward`方法，分别对神经网络之中每一层下定义，以及描述了层与层之间的连接关系，其中relu为激活函数

其次需要对所声明的类进行实例化：

```python
net =Net()
optimizer = torch.optim.Adam(net.parameters(),lr=0.05)
loss_func = torch.nn.MSELoss()
```

其中optimizer为优化器，pytorch提供了SGD、Adam等优化器可供选择，优化器方法之中需要`net.parameters()`作为参数，其中的lr为学习率

loss_func为损失函数，此处所选择的是mean square error作为损失函数

最后是训练的过程：

```python
loss_all=[]
for t in range(300):
    pred = net(x)
    loss = loss_func(pred,y) #计算损失
    optimizer.zero_grad() #清空优化器中原有的梯度
    loss.backward() #将损失进行反向传播
    optimizer.step() #更新参数
    loss_all.append(loss.data)
plt.plot(loss_all,"r-") #绘图
plt.show()
```

loss的变化趋势如下图所示：

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202208252229087.png" alt="image-20220825222949037" style="width: 33%;" />

以下为拟合的曲线与原曲线的对照：

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202208252231556.png" alt="image-20220825223109515" style="width: 33%;" />

### pytorch数据集的加载

pytorch提供了Dataset类，可以方便地实现数据集的加载：

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class MyDataset(Dataset):
    def __init__(self):
        pass #该方法负责数据的预处理
    def __getitem__(self,index):
        pass #该方法返回一组数据x和对应的标签y
        #return x,y
    def __len__(self):
        pass #该方法返回整个数据集的长度
        #return len
dataset = MyDataset()
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=1)
```

注意在使用Dataset时，在windows环境之中，会产生报错，解决方法是将代码主逻辑放入if语句之中：

```python
if __name__ == '__main__':
    #主逻辑
```

以下为实例

```python
class MyDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter='',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
    def __getitem__(self,index):
        return self.x_data[index],self.t_data[index]
    def __len__(self):
        return self.len
```

以及进行训练的实例：

```python
for epoch in range(100):
    for i,data in enumerate(train_loader, 0): #train_loader会将数据集始终的数据自动转化为tensor
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred,labels) #损失函数
        #print(epoch, i , loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 多分类与softmax

pytorch提供了CrossEntropyLoss，其中可以对神经网络的输出结果进行归一化，即转换为和为1的概率分布的形式，即Softmax，其使用方法为：

```python
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(y_pred,y)
```

其中y_pred为经过softmax的预测矩阵，y为经过one-hot编码的分类标签

### 最后
该项目之中分别采用了分batch的训练方式以及将数据集从头到尾遍历过一次的训练方式，实际尝试结果表明后者优于前者