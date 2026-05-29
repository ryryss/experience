# 2022
## 深度学习
机器学习中的关键组件: <br>可以用来学习的数据（data）<br>如何转换数据的模型（model） <br>一个目标函数（objective function）用来量化模型
的有效性 <br>调整模型参数以优化目标函数的算法（algorithm）

数据集由一个个样本（example, sample）组成，每个样本由一组称为特征（features，或协变量（covariates））的属性组成.  <br>预测的是
一个特殊的属性，它被称为标签（label，或目标（target））<br>

我们需要定义模型的优劣程度的度量，这个度量在大多数情况是“可优化”的，这被称之为目标函数（objective function）

过拟合？


深度学习流程： <br>
1、输入 随机搞个w 或者多个w <br>
2、使用w计算分数，卷积神经网络同，前向传播 <br>
3、激活函数 <br>
4、得到预测结果（Prediction）<br>
5、计算损失函数（Loss Function）<br>
6、反向传播（Backpropagation）<br>
7、梯度下降 / 优化器更新参数<br><br>

# 2026
20260527 开始AI学习
<br><br>

## PyTorch
让我们使用一个简单的demo去理解PyTorch框架如何使用<br>
https://gitee.com/kongfanhe/pytorch-tutorial<br>
```
#定义全连接层，第一个参数是输入维度, 第二个参数是神经元数量, 28表示输入训练材料的size
torch.nn.Linear(28*28, 64) 

#激活函数 线性转非线性
torch.nn.functional.relu

#softmax + nll_loss是经典组合, 还有很多种loss
#输出转频率 归一化的意思
output = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
# 衡量错误 输出越小越好 L = −logp(y)
torch.nn.functional.nll_loss(output, y)

# 找到最大概率类别 本例中类别即某个数字
torch.argmax(output)

# 优化器 更新权重用
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# 清空梯度
net.zero_grad()

# 自动求导反向传播 根据梯度更新参数
loss.backward()
optimizer.step()
```
<br>

## diffusion


扩散过程的核心公式：

$$
x_t = \sqrt { \alpha_t }x_0 + \sqrt {1 - \alpha_t } \epsilon
$$

其中： 
- $x_0$ ：原始图片
- $x_t$ ：第 $t$ 步后的带噪图片
- $\alpha_t$ ：当前 timestep 的噪声控制参数 
- $\epsilon$ ：高斯噪声（Gaussian Noise）


含义：当 $t$ 增大时 $\sqrt {1-\alpha_t}\epsilon$ 的占比会越来越大. 最终 $x_t$ 会逐渐接近纯随机噪声.<br><br>

Diffusion 模型训练的目标, 不是直接预测原图，而是预测噪声：
$$\epsilon_\theta(x_t, t)$$
其中：
- $\epsilon_\theta$ ：神经网络预测的噪声
- $x_t$ ：带噪图片
- $t$ ：当前 timestep

训练损失函数： $$L = \left \| \epsilon - \epsilon_ \theta(x_t,t)\right\| ^2$$
这是一个 MSE（均方误差）损失. 
含义：
- $\epsilon$ ：真实加入的噪声
- $\epsilon_\theta(x_t,t)$ ：模型预测的噪声

训练目标： 让模型预测的噪声尽可能接近真实噪声.
<br><br>

## Tensor
<br><br>