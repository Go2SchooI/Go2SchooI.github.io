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

- 具身（Embodied）：表示该**模型接入了机器人**，具有了身体。

- 多模态（Multimodal）：表示该模型的输入是多模态的，包括**文本**（textual input encodings）、**视觉**（visual）、**连续状态估计**（continuous state estimation）。

- 语言（Language）：表示该模型的**输出只有文本**。虽然说输出只能是文本，但是这个输出不再限制于自然语言。因为**代码本身也是文本，**所以模型的输出可以是一段代码，代码执行完之后可以是一段机器人可识别的指令；也可以模型直接输出机器人可识别的指令；这样模型的输出结果就可以操作机器人。

  

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

PaLM-E的主要架构思想是将连续的、具体化的观察（如图像、状态估计或其他传感器模态）注入预训练的语言模型的语言嵌入空间。这是通过将连续观察结果编码为与语言标记的嵌入空间具有相同维度的向量序列来实现的。因此，连续信息以类似于语言标记的方式注入到语言模型中。**PaLM-E**是**一种仅用于解码器的LLM**，它在给定前缀或提示的情况下自动生成文本补全。称论文的模型为**PaLM-E**，因为论文使用**PaLM**（Chowdhery等人，2022）作为预训练语言模型，并使其具体化。

**PaLM-E**的输入包括文本和（多个）连续观察。对应于这些观察的多模态标记与文本交错，以形成多模态句子。这样一个多模态句子的例子是问：**＜img 1＞和＜img 2＞之间发生了什么？**其中＜img i＞表示**图像的嵌入**。PaLM-E的**输出是**模型**自回归生成的文本**，它可以是问**题的答案**，也可以是PaLM-E**以文本形式生成的一系列决策**，**这些决策应由机器人执行**。当PaLM-E负责制定决策或计划时，论文假设存在一个低级策略或计划器，可以将这些决策转化为低级行动。先前的工作讨论了训练此类低级政策的各种方法（Lynch&Sermanet，2020；Brohan等人，2022），论文直接使用这些先前的方法而不进行修改。



该模型的基座是之前 google 发布的预训练模型 PaLM，然后接上机器人，也就是具身（Embodied），所以该模型的名字为 PaLM-E（PaLM + Embodied）。既然基座是 PaLM 模型，那么该模型就是 Decoder 模型。

模型 PaLM-E 的输入有三种类型：文本、图像、连续状态（来自于机器人的各种传感器的观测结果）。输入中的连续状态和输入中的文本一样，映射到相同维度的向量空间中之后输入到模型中，至于如何映射在后面进行说明。在输入模型时文本、图像、连续状态这三部分的顺序是不固定的，有可能交替出现，比如以如下这种方式：

```text
Q: What happened between <img 1> and <img 2>?
```

该模型的输出是仅有文本输出，这些文本输出除了可以是之前的经典的图像语言模型的任务的输出，还可以表示机器人的底层决策/计划，然后用于控制机器人的底层行为。在本文中重点研究的就是用模型的输出控制机器人的决策。

## **参考文献**

[1] [PaLM-E: 具身多模态语言模型（Embodied Multimodal Language Model） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/615879292)