## 20260528
根据此视频，了解Pytorch框架 <br>
【10分钟入门神经网络 PyTorch 手写数字识别】 https://www.bilibili.com/video/BV1GC4y15736/?share_source=copy_web&vd_source=931351c15183ddc93605f0c1d3208ca8<br><br>

输入5x5的灰度图片，使用神经网络节点公式<br>

$$
x^{k+1}_j = \sum_i f(a^k_{i,j}, x^k_i + b^k_j)
$$
其中：
* $x^k_i$ 表示第 $k$ 层的第 $i$ 个节点
* $a^k_{i,j}$ 表示从第 $k$ 层节点 $i$ 到第 $k+1$ 层节点 $j$ 的权重
* $b^k_j$ 表示第 $k+1$ 层节点 $j$ 的偏置项
* f是激活函数

套上激活函数, 就变成了神经网络前向传播公式

$$
x^{k+1}_j= \sigma \left (\sum_i a^k_{i,j} , x^k_i+b^k_j \right)
$$

其中 $\sigma$ 表示激活函数


<br>
在本例中公式的输出是概率值, 即10种数字的概率<br>
而训练的目的是, 基于梯度下降、ADAM算法等其他算法, 在不断的训练过程中得到a, b的值, 提高准确率
<br><br>

## 20260529
diffusion 扩散是如何训练的