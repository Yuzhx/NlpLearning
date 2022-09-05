## Related Works

### RNN model

沿输入和输出序列的符号位置计算因子。将位置与计算时间中的步骤对齐，生成隐藏状态ht序列，ht-1会作为ht的输入，这种固有的顺序性质排除了训练示例内的并行化，因子分解技巧[21]和条件计算[32]显著提高了计算效率和模型性能。但是约束仍然存在。

### CNN model

扩展神经GPU[16]、ByteNet[18]和Convs2[9]等使用卷积神经网络作为基本构建块，并行计算所有输入和输出位置的隐藏表示，以实现减少顺序计算的目标。在这些模型中，将来自两个任意输入或输出位置的信号关联起来所需的操作数量随着位置之间的距离而增加。convs2：线性，ByteNet：对数。导致学习远距离位置之间的相关性更加困难

### Attention mechanisms 

可以在不考虑特征之间的依赖关系，而不考虑它们在输入或者输出序列之中的距离。然而，在除少数情况外的所有情况下[27]，这种注意力机制与循环网络结合使用。与此不同，Transformer是一种避免重复的模型架构，而完全依赖于注意机制来绘制输入和输出之间的全局依赖关系。

### Self Attention mechanisms 

又称为内部注意，将单个序列的不同位置连接起来，以计算序列的表示。

## Transformer

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209051004688.png" alt="image-20220905100411572" style="zoom:67%;" />

编码器将输入序列x1~xn映射到z1~zn的序列下。给定z，编码器每次生成一个y，y1~yn作为输出序列。即在生成下一个符号时，消耗先前生成的符号作为额外输入。Transformer的架构如上图所示。

### Encoder

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209051012684.png" alt="image-20220905101223627" style="zoom:67%;" />

编码器由N＝6个相同层的堆叠组成。每个层有两个子层。第一种是多头自注意力机制，第二种是简单的、按位置完全连接的前馈网络。在经过子层之后，我们还进行了剩余连接与层归一化（Add & Norm）

#### batch-norm与 layer-norm

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209042308885.png" alt="image-20220904230813807" style="zoom: 25%;" />

蓝色为batch-norm，黄色为layer-norm，使所取出来的一个平面内的特征值变为均值为0，方差为1。layer-norm用的更广泛的原因：在时序模型里面，同一个batch之中不同样本的长度会发生变化。在算均值时，对于batch-norm，在样本长度变化较大时，其均值和方差的抖动会比较大，并且batch-norm之中全局的均值方差未能起到很好的指导意义。

### Decoder

解码器之中引入了第三个子层，同样是多头注意力机制。

自回归：当前的输出的输入集是之前时刻的输出。引入了mask，在预测t时刻的输出时，不能看到t时刻之后的输入

### Attention

output为value的一个加权和，其权重为key和query的相似度（在不同注意力机制下有不同的算法，原文采用：Scaled Dot-Product算法：两个向量做内积，其余弦值越大则相似度越高，如下图所示）

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209051050661.png" alt="image-20220905105018592" style="zoom: 50%;" />

![image-20220904232202656](https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209042322704.png)

行列转换关系如下图所示

 <img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209042328547.png" alt="image-20220904232801507" style="zoom:50%;" />

当dk比较大的时候（向量比较长），此时相对差距会变大

### Multi-Head Attention

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209051542631.png" alt="image-20220905154217685" style="zoom:67%;" />

多头注意力机制，在自注意力机制（对于输入的embedded矩阵，只使用一组Wq，Wk，Wv进行变换得到一组Query，Keys，Values，经过Matmul得到一组z矩阵）的基础之上，采取了多组（h组）Wq，Wk，Wv，最终得到多组Query，Keys，Values，每一组Q、K、V分别经过Matmul，最终得到多组z矩阵，最后将得到的多个Z矩阵进行拼接。本文之中所选择的是h=8（每个头的投影维度为dmodel/h）

相比于经典的自注意力机制，多头注意力机制可以进行学习，以产生多种不同的匹配模式

### 注意力机制的三种使用方法

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209051004688.png" alt="image-20220905100411572" style="zoom:67%;" />

在本模型之中，注意力机制（多头）被使用了三次，分别对应图中的三个橙色的模块：

左下角为对输入矩阵进行处理的多头自注意力机制，之所以称为自注意力机制，因为其qkv都相同，都是输入矩阵。实际上其输出为分别计算每一个向量的加权和，其权重为和所有向量的相似度（一句话之中词与其他词之间的关联程度）

右下角为解码器之中的具备掩码的多头自注意力机制，即对于一个句子，只计算词与其前面的词的关联程度，通过mask将词与之后的词的权重设为0

右上角为多头注意力机制，其key和value来源于encoder的输出，query来源于decoder的前一层的输出

### point-wise feed forward network

其本质是一个mlp，其只作用在最后一个维度（词嵌入向量的维度），等效于单隐藏层的mlp，中间一层将输入扩大四倍之后再缩小

### Embedding

对于任何一个词，学习一个dmodel长度的向量来表示它，为了防止维度过大导致值过小，所以会×根号dmodel

### Positional Encoding

考虑词在句子之中的不同位置，引入了一个位置编码

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209052027175.png" alt="image-20220905202709137" style="zoom:50%;" />

 从而防止单词在句子之中的位置信息被忽略

### 模型之中不同层的比较

![image-20220905204124372](https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/202209052041471.png)

