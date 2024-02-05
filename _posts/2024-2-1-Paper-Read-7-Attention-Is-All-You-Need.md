---
layout: post
title:  "Paper Read 7 - Attention Is All You Need"
date:   2024-2-1 1:18:00 +0800
tags:
- Embodied Intelligence
- NLP
toc:  true
math: true

---

The emergence of LLM has advanced the field of robotics. These models can learn rich linguistic knowledge and semantic representations by being pre-trained on **large-scale textual data**. These models can then be fine-tuned to adapt to specific tasks or domains. Therefore, in this post, I am going to document my learning process of **embodied intelligence**. Since I only know about the robotics field before, and the main body of the paper is in the field of NLP, I decided to record in my native Chinese language in order to make it easier for me to understand.

This series is expected to consist of a relatively small number of posts, with the short-term end goal of sustaining my reading of the **Google PaLM-E** article. 

## **Attention**

详细课程地址：[10.【李宏毅机器学习2021】自注意力机制 (Self-attention) (上)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1v3411r78R?p=1&vd_source=8a1fa40e08f3ead438b9bc465bd04915)， 课程笔记[Self-attention笔记](https://go2schooi.github.io/self_v7.pdf). 

简单的举例理解：[注意力机制的本质Self-Attention Transformer QKV矩阵_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1dt4y1J7ov/?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click&vd_source=8a1fa40e08f3ead438b9bc465bd04915)



## **Transformer**

详细课程地址：[13.【李宏毅机器学习2021】Transformer (下)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1v3411r78R/?p=4&spm_id_from=pageDriver&vd_source=8a1fa40e08f3ead438b9bc465bd04915)，课程笔记[**Transformer**笔记](https://go2schooi.github.io/seq2seq_v9.pdf)；

补充视频课程：[李沐Transformer论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.top_right_bar_window_history.content.click&vd_source=8a1fa40e08f3ead438b9bc465bd04915)



## **Notes**

### **RNN**

对于处理输入、输出不定长且存在上下文依赖的序列数据，类似DNN、CNN网络其效率较低，且无法解决依赖问题。对此我们需引入循环神经网络。（RNN， Recurrent Neural Network）

RNN的核心思想即是将数据按时间轴展开，每一时刻数据均对应相同的神经单元，且上一时刻的结果能传递至下一时刻。至此便解决了输入输出变长且存在上下文依赖的问题。循环神经网络可以看做是数据以链状结构展开

循环神经网络可以看做在做“加法”操作，即通过加法中的进位操作，将上一时刻的信息或结果传递至下一时刻，如下所示：

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-8f07087b76575626965ce42792ed51b6_720w.webp" alt="img" style="zoom: 80%;" />

RNN单元结构及其在时序上的展开如下图所示：

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-ba14fc72ab114f3e9bd021973ab87a67_720w.webp" alt="img" style="zoom:125%;" />

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-1bfe169fc1aa4b0aaf4a68eca8c49455_720w.webp" alt="img" style="zoom:80%;" />

可以看出RNN网络的输入输出均为序列数据，其网络结构本质上是一个小的全连接循环神经网络，只不过在训练时按时序张开。

RNN网络前向传播过程如下：

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-c4a780a986aa435339198f6afd1f2c9a_720w.webp" alt="img" style="zoom:80%;" />

RNN网络误差反向传播过程如下：

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-90c7b57dbedca67cbfa80c0db9152936_720w.webp" alt="img" style="zoom:125%;" />

RNN的训练过程是子网络沿时间轴的展开，其本质上仍在训练同一个子网络，因此在每次梯度下降时我们需要对同一个子网络的参数进行更新。常见的做法是在每一次参数的调整使均指向同一内存地址。

RNN存在优化困难的问题

训练时间长，梯度消失 $\prod_{i=k+1}^t\left(W^h\right)^T \operatorname{diag}\left[f^{\prime}\left(a_{i-1}\right)\right]<1$ ，出现情况较少，但对优化过程影响很大

梯度爆炸问题  $\prod_{i=k+1}^t\left(W^h\right)^T \operatorname{diag}\left[f^{\prime}\left(a_{i-1}\right)\right]>1$ ，大部分情况

对于梯度消失问题，由于相互作用的梯度呈指数减少，因此长期依赖信号将会变得非常微弱，而容易受到短期信号波动的影响，而对于梯度爆炸问题，我们一般采取梯度截断的方法。目前解决RNN网络长期依赖问题更为普遍的做法是引入LSTM单元。

### **LSTM**

LSTM与单一tanh循环体结构不同，其拥有三个特殊的“门”结构，它是一种特殊的循环体网络。

LSTM单元不仅接受此时刻的输入数据 $x_t$ 和上一时刻的状态信息 $h_{t-1}$ ，其还需建立一个机制能保留前面远处结点信息不会被丟失。具体操作是通过设计“门”结构实现**保留信息和选择信息**功能 (遗忘门、输入门)，每个门结构由一个sigmoid层和一个poinewise操作 (按位乘法操作) 构成。其中sigmoid作为激活函数的全连接神经网络层会输出一个0到1之间的数值，描述**当前输入有多少信息量可以通过这个结构**，其功能就类似于一扇门。

对于遗忘门，其作用是让循环神经网络 “忘记”之前没有用的信息。遗忘门会根据当前的输入 $x_t$和上一时刻输出 $h_{t-1}$ 决定哪一部分记忆需要被遗忘。假设状态 $c$ 的维度为 $n$ ，“遗忘门" 会根据**当前的输入** $x_t$ 和上**一时刻输出** $h_{t-1}$ 计算一个维度为 $n$ 的向量 $f=\operatorname{sigmoid}\left(W_1 x+W_2 h\right)$ ，其在每一维度上的值都被压缩在 $(0,1)$ 范围内。最后将**上一时刻的状态** $c_{t-1}$ 与 $f$ 向量按位相乘，在 $f$ 取值接近0的维度上的信息就会被 “遗忘”，而 $f$ 取值接近1的维度上的信息将会被保留

在循环神经网络 “忘记”了部分之前的状态后，它还需要从**当前的输入补充最新的记忆**。这个过程就是通过输入门完成的。输入门会根据 $x_1$ 和 $h_{t-1}$ 决定哪些信息**加入到状态** $c_{t-1}$ 中**生成新的状态** $c_t$ 。输入门和需要写入的新状态均是由 $x_t$ 和 $h_{t-1}$ 计算产生。

LSTM结构在计算得到新的状态 $c_t$ 后需要**产生当前时刻的输出**，这个过程是通过输出门完成的。输出门将根据**最新的状态** $c_t$ 、**上一时刻的输出** $h_{t-1}$ 和**当前的输入** $x_t$ 来决定该时刻的输出 $h_t$。通过遗忘门和输入门的操作循环神经网络LSTM可以更加有效地决定哪些序列信息应该被遗忘，而哪些序列信息需要长期保留，其结构如下图所示:

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-398d2b1fb1a43e5a7afe6c692a1f1a63_720w.webp" alt="img" style="zoom:80%;" />

![img](https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-15c32d4aafcf110773863ec6b60b0532_720w.webp)

### **GRU**

LSTM结构如上图所示，可以看出其结构较为复杂，这也引起了研究人员的思考：LSTM结构中到底那一部分是必须的？还可以设计哪些结构允许网络动态的控制时间尺度和不同单元的遗忘行为？对此设计GRU单元(Gated recurrent unit)，以**简化**LSTM，如下所示：

![img](https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-ddb90c9a2a7769ad03f50e9e84d497bc_720w.webp)

如上图所示，GRU与LSTM最大的区别是其**将输入门和遗忘门合并为更新门**（更新门决定隐状态保留放弃部分）。

### **Encoder-Decoder**

虽然LSTM确实能够解决序列的长期依赖问题，但是对于**很长的序列**（长度超过30），LSTM效果也难以让人满意，这时我们需要探索一种更有效的方法，即**注意力机制**（attention mechanism）。在介绍注意力机制前，我们先了解一种常用的框架：**Encoder-Decoder**框架。

在上文的讨论中，我们均考虑的是输入输出序列等长的问题，然而在实际中却大量存在**输入输出序列长度不等**的情况，如机器翻译、语音识别、问答系统等。这时我们便需要设计一种**映射可变长序列至另一个可变长序列的RNN网络结构**，Encoder-Decoder框架呼之欲出。

Encoder-Decoder框架是**机器翻译**（Machine Translation）模型的产物，其于2014年在**Seq2Seq**循环神经网络中首次提出。在统计翻译模型中，模型的训练步骤可以分为预处理、词对齐、短语对齐、抽取短语特征、训练语言模型、学习特征权重等诸多步骤。而Seq2Seq模型的基本思想非常简单一一使用一个**循环神经网络读取输入句子**，将整个句子的信息**压缩到一个固定维度**（注意是固定维度，下文的注意力集中机制将在此做文章）的编码中；再使用**另一个循环神经网络读取这个编码**，将其**解压为目标语言的一个句子**。这两个循环神经网络分别称为编码器（Encoder）和解码器（Decoder），这就是 encoder-decoder框架的由来。如下图所示：

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-057f266dbfa98cd795b20f722661a78c_720w.webp" alt="img" style="zoom:80%;" />

Decoder：根据 $x$ 的**中间语义表示** $c$ 和**已经生成**的 $y_1, y_2, \ldots, y_{i-1}$ 来**生成 $i$ 时刻**的 $y_i, y_i=g\left(c, y_1, y_2, \ldots, y_{i-1}\right)$ 。解码器部分的结构与一般的语言模型几乎完全相同: 输入为单词的词向量，输出为softmax层产生的单词概率，损失函数为log perplexity。事实上，解码器可以理解为一个以输入编码为前提的语言模型。语言模型中使用的一些技巧，如共享softmax层和词向量的参数，均可以直接应用到 Seq2Seq模型的解码器中。

Encoder：对输入序列 $x$ 进行编码，通过**非线性变换**转化为中间语义表示 $c, c=F\left(x_1, x_2, \ldots, x_m\right)$ 。编码器部分网络结构更为简单。它与解码器一样拥有**词向量层和循环神经网络**，但是由于在编码阶段并未输出最终结果，因此不**需要softmax层**。

Encoder-Decoder是一个十分通用的计算框架。此外，Encoder-Decoder框架其本质是**实现直观表示**（例如词序列或图像）和**语义表示**之间来回映射。故通过该框架我们可以使用来自一种模态数据的编码器输出作为用于另一模态的解码器输入，以实现将一种模态转换到另一种模态的系统。正因为这个强大的功能，Encoder_Decoder框架以应用于机器翻译，图像生成标题等众多任务中。

### **Attention Mechanism**

Attention Mechanism最早引入至自然语言中是为解决机器翻译中随**句子长度增加其性能显著下降的问题**，现已广泛应用于各类序列数据的处理中。

**机器翻译问题**本质上可以看做是**编码解码问题**，即将句子编码为向量然后解码为翻译内容。早期的处理方法一般是将句子拆分为一些小的片段分别单独进行处理，深度学习通过构建一个大型的神经网络对同时考虑整个句子内容，然后给出翻译结果。其网络结构主要包括**编码和解码两个部分**，如双向RNN网络。

**编码器**将一个句子**编码为固定长度的向量**，而解码器则负责进行解码操作。然而实验表明这种做法随着句子长度的增加，其性能将急剧恶化，这主要是因为**用固定长度的向量去概括长句子的所有语义细节十分困难**。为克服这一问题，Neutral Machine Translation by Jointly Learning to Align and Translate文中提出每次读取整个句子或段落，通过**自适应的选择**编码向量的**部分相关语义细节片段**进行解码翻译的方案。即对于整个句子，每次选择相关**语义信息最集中**的部分同时考虑**上下文信息**和**相关的目标词出现概率**进行翻译。注意力集中机制（Attention Mechanism）（本质上是加权平均形成上下文向量），这在长句子的处理中得到了广泛的应用。

注意力机制的主要亮点在于对于**seq2seq**模型中**编码器将整个句子压缩为一个固定长度的向量** $c$ ，而当句子**较长时**其很难保存足够的语义信息，而**Attention**允许解码器根据当前**不同的翻译内容**，查阅输入句子的**部分不同的单词或片段**，以提高每个词或者片段的翻译精确度。

一个自我关注模块接受n个输入并返回n个输出。这个模块中发生了什么?用外行人的话来说，自我注意机制允许**输入相互作用**(“自我”)，并**发现他们应该更关注谁**(“注意”)。这些输出是这些交互作用和注意力分数的总和。



具体做法为解码器在每一步的解码过程中，将**查询编码器的隐藏状态**。对于整个输入序列计算**每一位置** (每一片段)与**当前翻译内容**的**相关程度，即权重**。再根据这个权重对**各输入位置的隐藏状态**进行**加权平均**得到 "context" 向量 (Encoder-Decoder框架向量 $c$ )，该结果包含了**与当前翻译内容最相关的原文信息**。同时在**解码下一个单词**时，将**context作为额外信息输入**至RNN中，这样网络可以时刻读取原文中最相关的信息，而不必完全依赖于上一时刻的隐藏状态。Attention本质上是通过加权平均，计算可变的上下文向量 $c$ 。

![img](https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-4220b36ee817f732e9b133f86fca71dd_720w.webp)

![img](https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-a2768d317695237f8858e738edc2807e_r.jpg)

![img](https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/v2-babcc3f177f9b39600ec2251232e058b_720w.webp)

其中，第 $j$ 时刻的 context $j$ 计算如下:
$$
\begin{aligned}
& \alpha_{i j}=\frac{\exp \left(e\left(h_i, s_j\right)\right)}{\sum_i \exp \left(e\left(h_i, s_j\right)\right)} \\
& e(h, s)=U \tanh (V h+W s) \\
& \text { context }_j=\sum_i \alpha_{i, j} h_i
\end{aligned}
$$

上式中， $U, V, W$ 为模型参数。 $h_i$ 表示编码器在第 $i$ 个单词上的输出， $s_j$ 为编码器预测第 $j$个单词的状态， $\alpha$ 为通过Softmax计算的权值， $e(h, s)$ 为计算原文各单词与当前解码器状态的 "相关度" 函数，其构成了包含一个隐藏层的全连接神经网络。

## **References**

[白话机器学习-Attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/508089056)

[白话机器学习-Self Attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/508422850)

[白话机器学习-Encoder-Decoder框架 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/507798134)

[从RNN、LSTM到Encoder-Decoder框架、注意力机制、Transformer - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/50915723)