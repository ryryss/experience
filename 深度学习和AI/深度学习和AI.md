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
```
# https://gitee.com/kongfanhe/pytorch-tutorial

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
如何让计算机绘制一张图片? 假设输入是纯噪点, 然后再一步一步降噪，直至还原成一张精美的图片<br>
如何降噪? 降噪的依据和条件的什么? <br>
使用带噪声的图片不断训练机器, 每训练一次就加一点噪声, 让机器记住噪声条件<br>
需要生成图片时, 则执行上述的相反操作, 即可通过不断的降噪, 从而生成图片<br><br>

推导过程就不管了直接看, 
训练扩散过程的核心公式：

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

$$
\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i
$$

$$
\alpha_t = 1-\beta_t
$$
其中： 
- $x_0$ ：原始图片
- $x_t$ ：第 $t$ 步后的带噪图片
- $\alpha_t$ ：当前 timestep 的噪声控制参数 
- $\epsilon$ ：高斯噪声（Gaussian Noise）


含义：$\beta_t$ 由 schedule 得到, 当 $t$ 增大时 $\sqrt {1-\bar{\alpha}_t}\epsilon$ 的占比会越来越大. 最终 $x_t$ 会逐渐接近纯随机噪声.<br><br>

预测噪声：
$$\epsilon_\theta(x_t, t)$$
其中：
- $\epsilon_\theta$ ：神经网络预测的噪声
- $x_t$ ：带噪图片
- $t$ ：当前 timestep

训练损失函数： $$L = \left \| \epsilon - \epsilon_ \theta(x_t,t)\right \| ^2$$
这是一个 MSE（均方误差）损失. 
含义：
- $\epsilon$ ：真实加入的噪声
- $\epsilon_\theta(x_t,t)$ ：模型预测的噪声

训练目标： 让模型预测的噪声尽可能接近真实噪声.
<br><br>
结合minDiffusion项目的代码去理解上述公式
```
# https://github.com/cloneofsimo/minDiffusion

# 未知数 beta_t, 由 Linear Noise Schedule 线性噪声计划 得到
for k, v in ddpm_schedules(betas[0], betas[1], n_T).items()

# ddpm_schedules 部分源码如下
beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
alpha_t = 1 - beta_t
log_alpha_t = torch.log(alpha_t)
alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

# 最终得到 alphabar_t 返回结果并保存起来
return {
    "alpha_t": alpha_t,  # \alpha_t
    "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
    "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
    "alphabar_t": alphabar_t,  # \bar{\alpha_t}
    "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
    "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
    "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
}
```
```
# 知道变量的来源之后, 就可以使用公式进行训练了
# 以下是 forward 函数的部分源码
# 首先给输入的图片随机加噪到随机时间步
_ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)

# 每像素一个高斯噪声
eps = torch.randn_like(x)  # eps ~ N(0, 1)

# 加噪并返回loss
x_t = (
    self.sqrtab[_ts, None, None, None] * x
    + self.sqrtmab[_ts, None, None, None] * eps
)  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
# We should predict the "error term" from this x_t. Loss is what we return.

# criterion的来源是 criterion: nn.Module = nn.MSELoss()
return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

# 接着优化网络, 继续迭代训练
loss.backward()
optim.step()

# 使用推理模式检查训练质量
ddpm.eval()
with torch.no_grad():
    xh = ddpm.sample(16, (1, 28, 28), device)
    grid = make_grid(xh, nrow=4)
    save_image(grid, f"./contents/ddpm_sample_{i}.png")
```
```
# sample 函数的内容: 从纯噪声出发, 一步一步去噪, 最终得到生成的图片

# 完全随机噪声
x_i = torch.randn(n_sample, *size)

# 使用网络预测噪声, 并计算原图
for i in range(self.n_T, 0, -1):
    eps = self.eps_model(x_i, i / self.n_T)
    x_i = (
        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
        + self.sqrt_beta_t[i] * z
    )
return x_i 
```
## Transformer

<br><br>

## Tensor
<br><br>