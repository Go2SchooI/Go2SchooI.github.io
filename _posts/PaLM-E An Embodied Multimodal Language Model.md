# PaLM-E: An Embodied Multimodal Language Model

由于本人之前只对机器人领域有所了解，而论文的主体是NLP领域，因此为了便于自己理解，决定用母语中文来记录研读过程。

## **前言**

这篇论文的名字是 "具身多模态语言模型"（Embodied Multimodal Language Model），先来分析一下这个名字是啥意思：

- 具身（Embodied）：表示该**模型接入了机器人**，具有了身体。
- 多模态（Multimodal）：表示该模型的输入是多模态的，包括**文本**（textual input encodings）、**视觉**（visual）、**连续状态估计**（continuous state estimation）。
- 语言（Language）：表示该模型的**输出只有文本**。虽然说输出只能是文本，但是这个输出不再限制于自然语言。因为**代码本身也是文本，**所以模型的输出可以是一段代码，代码执行完之后可以是一段机器人可识别的指令；也可以模型直接输出机器人可识别的指令；这样模型的输出结果就可以操作机器人。







## **参考文献**

[1] [PaLM-E: 具身多模态语言模型（Embodied Multimodal Language Model） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/615879292)