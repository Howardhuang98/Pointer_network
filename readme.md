# Pointer networks Tensorflow2

原文：https://arxiv.org/abs/1506.03134  
数据：http://goo.gl/NDcOIG  
仅供参考与学习，内含代码备注  

## 性能表现
|节点数目|模型预测路径长度-最佳路径长度|超参数|
|----|----|----|
|5|0.033425577472687575|adam优化器，LSTM128个隐藏层|
|5|0.02872023683121583|SGD，LSTM256个隐藏层|

![image-model_accuracy](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/model_accuracy.png)
![image-model_loss](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/model_loss.png)

## 环境
tensorflow==2.6.0  
tqdm  
matplotlib   
numpy  

# 《pointer networks》阅读笔记
应用场景：

文本摘要，凸包问题，Roundelay 三角剖分，旅行商问题 

其中包括一些Latex，github无法渲染，所以建议clone下来用Typora查看。
## abstract

本文提出一种新的网络结构：输出序列的元素是与输入序列中的位置相对应的离散标记。

> an output sequence with elements that are discrete tokens corresponding to positions in an input sequence. 

这种问题目前可以被一些现有的方法解决：sequence-to-sequence， neural turing machines。但是这些方法不是特别适用。

本文解决的问题是sorting variable sized sequences，以及各种组合优化问题。本模型使用attention机制来解决变化尺寸的输出。

## intro

RNN模型的输出维度是固定的，sequence-to-sequence模型移除了这一个限制，通过用一个RNN把输入映射为一个embedding，又用一个RNN把embedding映射到输出序列。

但是这些sequence-to-sequence 方法都是固定大小的词汇表。

> 例如词汇表中只存在A,B,C。那么输入
>
> 1,2,3 ---->  A,B,C
>
> 1,2,3,4 ---->  A,B,C,A

本文提出的框架适用于**输出的词汇表大小**取决于**输入问题的大小**。

![image-20211105133740833](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/image-20211105133740833.png)

![image-20211105134312635](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/image-20211105134312635.png)

左图：seq-2-seq

蓝色RNN，输出一个向量。

紫色RNN，利用概率的链式法则，输出一个固定维度。

本文的贡献如下：

1. 提出一种新的结构，称为指针网路。简单且高效
2. 良好的泛化性能
3. 一个TSP近似求解器

## Models

### sequence-to-sequence 模型

训练数据为：
$$
(P,C^P)
$$
其中，$\mathcal{P}=\left\{P_{1}, \ldots, P_{n}\right\}$，是n个向量。$\mathcal{C}^{\mathcal{P}}=\left\{C_{1}, \ldots, C_{m(\mathcal{P})}\right\}$ ，n个对应的结果，$m(\mathcal{P})\in [1,n]$ 。传统的sequence-to-sequence的$\mathcal{C}^{\mathcal{P}}$是固定大小的，但是要提前给定。本文的$\mathcal{C}^{\mathcal{P}}$为n，根据输入改变。

如果模型的参数记为$\theta$，神经网络模型表达为：
$$
p(C^P|P,\theta)
$$
使用链式法则，写为：
$$
p\left(\mathcal{C}^{\mathcal{P}} \mid \mathcal{P} ; \theta\right)=\prod_{i=1}^{m(\mathcal{P})} p_{\theta}\left(C_{i} \mid C_{1}, \ldots, C_{i-1}, \mathcal{P} ; \theta\right)
$$
训练阶段，最大似然概率：
$$
\theta^{*}=\underset{\theta}{\arg \max } \sum_{\mathcal{P}, \mathcal{C}^{\mathcal{P}}} \log p\left(\mathcal{C}^{\mathcal{P}} \mid \mathcal{P} ; \theta\right)
$$
input sequence的末端加一个$\Rightarrow$，代表进入生成阶段，$\Leftarrow$代表结束生成阶段。

推断：
$$
\hat{\mathcal{C}}^{\mathcal{P}}=\underset{\mathcal{C}^{\mathcal{P}}}{\arg \max } p\left(\mathcal{C}^{\mathcal{P}} \mid \mathcal{P} ; \theta^{*}\right)
$$
### content based input attention

 对于attention机制，请查看《Neural Machine Translation By Jointly Learning To Align And Translate》阅读笔记。

对于LSTM RNN
$$
\begin{aligned}
u_{j}^{i} &=v^{T} \tanh \left(W_{1} e_{j}+W_{2} d_{i}\right) & j \in(1, \ldots, n) \\
a_{j}^{i} &=\operatorname{softmax}\left(u_{j}^{i}\right) & j \in(1, \ldots, n) \\
d_{i}^{\prime} &=\sum_{j=1}^{n} a_{j}^{i} e_{j} &
\end{aligned}
$$
对于这个传统的attention机制，可以看到$u^{i}$, 是一个长度为$n$的向量。

这样的话，在解码器的每一个时间步迭代都会得到一个 n 长度的向量，可以作为指针，用于指向之前的 n 长度的序列。

### Ptr-Net

所以Ptr-Net计算公式写为：
$$
\begin{aligned}
u_{j}^{i} &=v^{T} \tanh \left(W_{1} e_{j}+W_{2} d_{i}\right) \quad j \in(1, \ldots, n) \\
p\left(C_{i} \mid C_{1}, \ldots, C_{i-1}, \mathcal{P}\right) &=\operatorname{softmax}\left(u^{i}\right)
\end{aligned}
$$
![image-20211111103159924](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/image-20211111103159924.png)

![image-20211111110334755](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/image-20211111110334755.png)

数据以 [Batch, time_steps, feature] 的形式进入编码器LSTM（绿色部分），在时间步上迭代$n$次以后，得到：

- n 个 e [batch, units], 可以合并写为 [batch, n, units]

- 最后一个时间步输出的 c [batch, units] 

进入到解码器LSTM（蓝色部分），输入为：

- 上次得到解码得到的的pointer，如果是第一次则为initial pointer
- 上次的状态d,c

pointer 如何得到？计算公式如下：
$$
\begin{aligned}
u_{j}^{i} &=v^{T} \tanh \left(W_{1} e_{j}+W_{2} d_{i}\right) \quad j \in(1, \ldots, n) \\
p\left(C_{i} \mid C_{1}, \ldots, C_{i-1}, \mathcal{P}\right) &=\operatorname{softmax}\left(u^{i}\right)
\end{aligned}
$$

## motivation and datasets structure

文章是为了解决三种问题，凸包，Delaunay Triangulation，旅行商问题。在此只对旅行商问题进行探讨。

### travelling salesman problem

给定一个城市列表，我们希望找到一条最短的路线，每个城市只访问一次，然后返回起点。此外，假设两个城市之间的距离在正反方向上是相同的。这是一个NP难问题，测试模型的能力和局限性。

数据生成：

卡迪尔坐标系（二维），$[0,1] \times[0,1]$

使用 Held-Karp algorithm 得到准确解，n最多为20。

A1,A2,A3为三种其他算法。A1，A2时间复杂度为$O\left(n^{2}\right)$，A3时间复杂度为$O\left(n^{3}\right)$。A3，Christofides algorithm 算法保证在距离最佳长度1.5倍的范围内找到解，详细信息查看原文参考文献。生成1M个数据进行训练。

![image-20211111111416012](%E3%80%8Apointer%20networks%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.assets/image-20211111111416012.png)

分析表格：

1. n=5的时候，性能都很好
2. n=10，ptr-net的性能比A1好
3. n=50的时候，无法超过数据集性能（因为ptr-net使用不准确的答案进行训练的）
4. 只用n少的训练，推广到大n情况，性能不太好。

对于n=30的情况，Ptr-net算法复杂度为$O(n \log n)$，远低于A1,A2,A3。却有相似的性能，说明可发展空间还是很大的。

