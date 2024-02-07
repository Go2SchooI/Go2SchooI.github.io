---
layout: post
title:  "Paper Read 8 - PaLM-E: An Embodied Multimodal Language Model"
date:   2024-2-7 23:30:00 +0800
tags:
- Embodied Intelligence
- NLP
toc:  true
math: true


---

Recently, when I saw the unmanned system of Westlake University's explanation of [Embodied Intelligence: Large Language Models Enabling Manipulators Planning and Control](https://zhuanlan.zhihu.com/p/648023912), it was mentioned that the **intelligent capability** brought by **LLM** makes robots smarter, more flexible, and capable of adapting to complex operation scenarios. So I read the paper of Google's PaLM-E mentioned in the article, and the record is as follows.

Since I only know about the robotics field before, and the main body of the paper is in the NLP field, I decided to record the reading process in Chinese to facilitate my understanding.

## **前言**

这篇论文的名字是 "具身多模态语言模型"（Embodied Multimodal Language Model），先来分析一下这个名字：

具身（Embodied）：表示该**模型接入了机器人**，具有了身体。

多模态（Multimodal）：表示该模型的输入是多模态的，包括**文本**（textual input encodings）、**视觉**（visual）、**连续状态估计**（continuous state estimation）。

语言（Language）：表示该模型的**输出只有文本**。虽然说输出只能是文本，但是这个输出不再限制于自然语言。因为**代码本身也是文本，**所以模型的输出可以是一段代码，代码执行完之后可以是一段机器人可识别的指令；也可以模型直接输出机器人可识别的指令；这样模型的输出结果就可以操作机器人。



## **摘要**

论文提出了具体化的语言模型，以将真实世界的连续传感器模态直接结合到语言模型中，从而建立单词和感知之间的联系。具体语言模型的**输入**是**多模态语句**，它们交织了**视觉**、**连续状态估计**和**文本输入编码**。

**结合预训练的大型语言模型**，对这些编码进行**端到端训练**，用于多个具体任务，包括**顺序机器人操作规划**、**视觉问题解答**和**图像视频字幕描述**。

论文的评估表明，**PaLM-E**可以在多个实施例上处理来自**各种观察模式**的各种体现推理任务，并且进一步表现出积极的迁移：该模型受益于跨互联网规模的语言、视觉和视觉语言领域的各种联合训练。

## **本文贡献**

- 提出并验证了可以通过将具身数据混合到多模态大语言模型的输入中来训练出一个通用的、可迁移的、多决策的 agent。
- 证实了虽然现有的 zero-shot 图像语言模型不能很好的解决具身推理问题，但是训练一个具有通用能力的、能够解决具身推理问题的图像语言模型是可行的。
- 关于如何训练上述具身多模态语言模型，本文提出了新颖的模型架构和训练策略。
- 本文提出的 PaML-E 除了能解决具身推理问题外，在常规的图像和文本任务领域效果也很好。
- 最后，本文证明了随着模型规模的增大，可以有效缓解模型在多模态微调时的灾难性遗忘问题。

## **模型结构**

该模型的基座是之前 google 发布的预训练模型 PaLM，然后接上机器人，也就是具身，所以该模型的名字为 PaLM-E（PaLM + Embodied）。既然基座是 PaLM 模型，那么该模型就是 **Decoder 模型**。

模型 PaLM-E 的输入有三种类型：文本、图像、连续状态（来自于机器人的各种传感器的观测结果）。输入中的连续状态和输入中的文本一样，映射到**相同维度的向量空间**中之后输入到模型中。在输入模型时文本、图像、连续状态这三部分的顺序是不固定的，有可能**交替出现**，比如以如下这种方式：

```text
Q: What happened between <img 1> and <img 2>?
```

其中＜img i＞表示**图像的嵌入**。

该模型的输出是**仅有文本输出**，这些文本输出可以是**问题的答案**，还可以是PaLM-E**以文本形式生成的决策/计划**，用于控制机器人的底层行为，论文假设存在一个低级策略或计划器，可以将这些决策转化为低级行动。在本文中重点研究的就是用模型的输出控制机器人的决策。

### 模型结构公式

**Decoder语言模型:**

最经典的GPT结构，给定前面序列的token，预测联合概率最大的下一个token。公式如下:

$$
p\left(w_{1: L}\right)=\prod_{l=1}^L p_{L M}\left(w_l \mid w_{1: l-1}\right)
$$

该公式中 $$p\left(w_{1: L}\right)$$ 是总共 $$L$$ 个token的联合概率， $$p_{L M}$$ 是语言模型。

**带有Prefix的Decoder语言模型:**

在经典的GPT模型的基础上，在每条文本数据前面添加上一段**prefix prompt**。这个prefix prompt 可以是离散的，也可以是连续的。公式如下:

$$
p\left(w_{n+1: L} \mid w_{1: n}\right)=\prod_{l=n+1}^L p_{L M}\left(w_l \mid w_{1: l-1}\right)
$$

在上述公式中，**位置 1 到位置 n 是prefix prompt**，从**位置 n+1 往后是输入的文本数据**。在训练时，prefix prompt部分不参与计算loss。

**文本 Token 的嵌入空间：**

定义一下符号，因为下一部分会需要将**从传感器获取到的连续状态**也映射到这个同维度的向量空间中。这里先将这个向量空间的符号定义明确，下一部分说明连续状态的映射时会更清晰。

使用 $$W$$ 表示所有的token集合，使用 $$w_i$$ 表示某一个token，使用 $$\mathcal{X} \in R^k$$ 表示整个嵌入空间，使用 $$\gamma$$ 表示词嵌入的映射，使用 $$x$$ 表示某个token的词嵌入向量。定义了这些之后那么词嵌入的过程可以表示为: $$x_i=\gamma\left(w_i\right) \in R^k$$

**连续状态映射到嵌入空间:**

传感器会观测到很多连续状态（多模态），这些如果想要输入到模型中，也需要将其转换为向量，在本文中是将其转为与**文本的token嵌入相同的向量空间中**。

使用 $$\mathcal{O}$$ 表示观测到的所有连续状态的集合，使用 $$O_j$$ 表示某个观测到的具体的连续状态，训练编码器 $$\phi: \mathcal{O} \rightarrow \mathcal{X}^q$$ 将 (连续) 观测空间 $$\mathcal{O}$$ 映射为 $$\mathcal{X}$$ 中 $$q$$ 个向量序列，编码后的向量空间还是 $$\mathcal{X}$$。然后将这些向量与普通嵌入文本标记交错，以形成LLM的前缀。这意味着前缀中的每个向量 $$x_i$$ 由单词标记嵌入器 $$\gamma$$ 或编码器 $$\phi_i$$ 构成:

连续状态的映射过程与文本token的映射过程的不同之处在于：**一个token**只映射为**一个向量**，**一个连续状态**可能需要映射为**多个向量**。比如观测到的一个连续状态可能是一段声音，只用一个向量不能很好的表示这段声音，就需要多个向量来表示。

定义了这些之后，那么文本和观测到的连续状态到嵌入向量的过程可以表示为:

$$
x_i= \begin{cases}\gamma\left(w_i\right) & \text { if } i \text { is a text token } \\ \phi_j\left(O_j\right)_i & \text { if } i \text { is a vector from } O_j\end{cases}
$$

举个例子更好理解，比如原始数据为如下例子（本文中的传感器都是图像数据，没有音频数据，这里只是举个例子）：

```text
请提取录音中的内容<这里是一段录音内容>。
```

经过编码后的嵌入向量序列为: $$\left(x_1, x_2, \ldots, x_9, x_{10}, x_{11}, x_{12}, x_{13}\right)$$ ，这个向量序列中各个向量的来源如下（假设这段录音内容被编码为了三个向量）:

- $x_1$ : 来自于"请"
- $x_2:$ 来自于"提"
- ...
- $x_9$ : 来自于"容"
- $x_{10}$ : 来自于<录音内容>
- $x_{11}$ : 来自于 <录音内容>
- $x_{12}$ : 来自于 <录音内容>
- $x_{13}$ : 来自于"。"

说明白了如何对连续状态进行编码，应该可以很容易的想到：虽然文本token和连续状态都被映射到了相同的向量空间中，但是做这个映射操作的模型肯定是不同的模型。关于token如何编码应该都很熟悉，需要说明如何对传感器观测到的信息做编码。

### 在机器人控制回路中具象输出

PaLM-E是一种生成模型，基于多模型句子作为输入**生成文本**。为了将模型的输出连接到实例。论文区分了两种情况。如果任务可以通过**仅输出文本来完成**，例如，在具体的问题回答或场景描述任务中，则**模型的输出被直接认为是任务的解决方案**。

或者，如果PaLM-E用于解决一个**具体的计划或控制任务**，它会生成一个文本来**调节低级命令**。特别是，假设可以使用一些（小的）词汇表来执行低级技能的策略，而PaLM-E的成功计划必须包含一系列此类技能。请注意，PaLM-E必须根据**训练数据和提示自行确定哪些技能可用**，并且不使用其他机制来约束或过滤其输出。尽管这些策略受语言限制，但它们无法解决长期任务或接受复杂指令。因此，PaLM-E被集成到一个**控制回路**中，在该回路中，机器人**通过低级策略执行其预测决策**，从而产生新的观察结果，如果需要，PaLME能够根据这些观察结果重新规划。从这个意义上讲，PaLME可以理解为一种高级策略，它对低级策略进行排序和控制。

## **传感器观测信息的编码策略**





## **参考文献**

[1] [PaLM-E: 具身多模态语言模型（Embodied Multimodal Language Model） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/615879292)