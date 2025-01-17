摘要— 最近，由于部署了强化学习（RL），人形机器人在执行具有挑战性的任务方面取得了显著进展，然而，人形机器人的固有复杂性，包括设计复杂的奖励函数和训练整个复杂系统的难度仍然是一个显著挑战。
为了克服这些挑战，在许多迭代和深入调查之后，我们精心开发了一款全尺寸人形机器人“Adam”，其创新的结构设计极大地提高了**模仿学习**过程的效率和效果。此外，我们还开发了一种基于**对抗性运动先验**的新型模仿学习框架，该框架不仅适用于Adam，还适用于一般的人形机器人。使用该框架，Adam可以在运动任务中展现前所未有的类人特征。我们的实验结果表明，所提出的框架使Adam能够在复杂的运动任务中实现与人类可比较的性能，这是人形机器人中首次使用人类运动数据进行模仿学习。更多视频演示，请访问我们的YouTube频道：https://www.youtube.com/watch?v=7hK2ySYBa1I



## introduction

近年来，人形机器人领域受到了广泛关注，许多研究机构和公司相继发布了前沿创新和研究成果，标志着该领域的快速发展和崛起。波士顿动力的Atlas机器人展示了类似跑酷的机动能力；特斯拉的Optimus和Figure公司的人形机器人从人类数据中学习，执行复杂的桌面操作任务；双足机器人Cassie及其人形版本Digit由电机驱动，并成功地跨越了各种地形；著名的四足机器人公司Unitree发布了他们的人形机器人产品H1；Apptronik开发了一款名为Apollo的人形机器人，完全由推杆电动机驱动；在通用人工智能领域享有盛誉的OpenAI收购了1X机器人公司，并提出了一项关于体现智能的发展计划。以上表明，人形机器人技术正在成为研究人员和公司的关键方向之一，掌握人形机器人的核心技术对于弥合数字通用人工智能和有形硬件之间的鸿沟至关重要。

传统的机器人控制算法通常依赖于**精确的数学模型**和**预定义的运动规划**，在过去对于四足、双足和人形机器人的运动任务表现出了非常高效的效果。值得注意的是，波士顿动力的Atlas和Spot机器人通过使用模型预测控制（MPC）算法在各种演示中展示了这些方法的有效性。然而，这些算法通常依赖于**对环境的准确建模**，这可能会在**稳健性和泛化能力方面引入重大挑战**，特别是在未知或动态变化的环境中，在这些环境中，传统控制算法的性能可能会显著下降，从而限制了它们在更广泛的应用场景中的实用性。此外，对准确建模的依赖性还需要高水平的专业知识来构建和维护这些模型，增加了开发和调试的复杂性。

传统的机器人控制算法在适应性、灵活性和用户友好性方面显著受到限制，即使它们在特定环境中表现出色，这也激励了研究人员探索替代方法，以克服这些障碍，设计更智能、更适应性更强的机器人控制策略。其中，基于深度神经网络的强化学习算法在腿式机器人的控制方面取得了有希望的结果。通过与环境的交互，深度强化学习算法能够自主发现执行复杂任务的有效策略，并且可以潜在地扩展到未知或动态变化的环境中，从而为机器人提供了前所未有的适应性和灵活性。

强化学习算法在各种腿式机器人中取得了显著进展，但在人形机器人领域的应用仍然缺乏足够的探索。这主要归因于以下因素：首先，大多数人形机器人价格昂贵且难以维护，这对于经费有限的研究机构来说是一个相当大的障碍；其次，深度神经网络的**可解释性问题**和深度强化学习训练过程中的**Sim2Real差距**使得难以将模型转移到实际应用中；最后，人形机器人的复杂性远远超过其他腿式机器人，这使得在其训练过程中设计奖励函数和训练策略更具挑战性。

为了解决这些挑战，在多次设计和深入探索后，我们推出了由电机关节驱动的人形机器人“Adam”，它在成本上具有明显的优势，而不是传统的液压驱动机器人，其模块化设计有助于实验中的维修，并进一步降低了维护成本。此外，我们的高性能执行器确保了机器人在其肢体上具有接近人类的运动范围。针对强化学习中设置复杂奖励函数的难度，我们采用了一种创新策略，即**使用人类运动数据来指导学习过程**。结合我们的模仿学习训练框架，Adam在实验中的运动任务表现令人印象深刻。

总的来说，我们介绍了一款全新的人形机器人Adam，并提出了一种新的方法论和实验验证，用于人形机器人的学习、适应和优化，为人形机器人领域的研究和开发开辟了新的道路。本文的贡献可以总结为以下三点：
1）我们开发并详细介绍了一款创新的仿生人形机器人Adam，其肢体不仅具有接近人类的运动范围，而且在成本和维护便利性方面具有足够的优势。
2）我们设计并验证了一种新的**全身模仿学习框架**，适用于人形机器人，有效解决了在人形机器人强化学习训练中遇到的**复杂奖励函数设置问题**，极大地缩小了Sim2Real差距，提高了人形机器人的学习能力和适应性。
3）为了解决复杂人形机器人强化学习控制算法中的Sim2Real挑战，我们在我们的框架中引入了大量的**交叉验证和反馈调整**步骤。我们不仅展示了机器人在执行复杂动作任务时高度接近人类的表现，还为未来人形机器人的运动学习和优化提供了新的视角和数据支持。

## related work

A. 腿式机器人的运动
腿式机器人的运动在实现人形机器人方面取得了显著进展，其引入了强化学习（RL）起到了关键作用。机器人Cassie通过使用周期参数化的奖励函数实现了广泛的行走和奔跑模式，甚至使用预先优化的步态创下了100米短跑的吉尼斯世界纪录。Jeon等人强调了基于潜在的奖励塑造在加速学习和增强腿式机器人运动稳健性方面的有效性。类似地，Shi等人通过引入辅助力学习计划，扩展了腿式机器人的能力，从而促进了在没有明确参考的情况下的灵活运动学习。HRP-5P人形机器人通过利用执行器电流的反馈实现了出色的双足行走，而Kim等人提出了一种基于扭矩的方法，有效地弥合了模拟训练和实际应用之间的差距。DeepMind的研究使一个微型人形机器人通过独特的师生蒸馏和自我对抗方法掌握了复杂的足球技能。此外，Digit人形机器人中使用基于注意力的转换器促进了更具适应性和多功能的运动模式。最近，Tang等人采用了对抗评论组件和特殊设计的Wasserstein距离，将运动从人类参考中迁移。然而，Adam具有更大的灵活性，总体更类似于人类，这导致了在实际机器人实验中的更好表现。

B. 从人类参考学习
人类以其先进的智能和多功能的运动能力展现了复杂的动作模式，体现了丰富的信息。通过从人类行为中获取的见解，可以极大地提高机器人的适应能力。传统的行为克隆方法依赖于手工编程，这被证明是耗时且不灵活的。此外，手动定义人形机器人的复杂和多功能的运动技能也带来了重大挑战。近年来，模仿学习（IL）策略日益受到关注，涉及**跟踪参考关节轨迹或提取的步态特征**。然而，这些跟踪技术通常**在单独的运动片段上操作**，在不同运动模式之间过渡时会产生不连续性。为解决这一限制，Peng等人引入了生成对抗性模仿学习（GAIL），提出了AMP和Successor ASE两种创新方法。这些方法使基于物理的角色能够执行客观任务，同时隐式地模仿来自广泛的非结构化数据集的多样化运动风格。AMP的变体已成功应用于学习敏捷的四足动物运动和适应地形的技能。此外，Tang等人引入了一种具有软边界约束的Wasserstein对抗性模仿系统，进一步增强了AMP方法的能力。为了促进参考运动向机器人的转移，许多作品引入了重新定位技术，考虑了原始骨架和几何一致性，实现了准确的动态建模和复杂的平衡控制器。

## PRELIMINARY

A. 人形机器人Adam的结构
本文使用了Adam的精简版本进行实验。Adam（Lite）全身配备了25个QDD（准直驱动）力控PND执行器，身高1.6米，重量60公斤。其腿部装配了四个QDD高灵敏度、高回驱动力的执行器，最大扭矩可达340N·m。手臂具有五个自由度，腰部具有三个自由度。这种完全模块化、高度可重复使用的设计采用了灵活的执行器，并配有高度仿生的躯干配置，使得Adam具有出色的机动性和适应性。整个身体采用了自主设计的全套解决方案，包括实时通信网络PDN（PND网络）和PND执行器。运动控制计算机采用第12代英特尔i7处理器（Intel NUC）和PND RCU（机器人控制单元）。PND RCU集成了所有执行器、电池管理系统（BMS）、功率管理，并配备了一个带网络管理功能的16端口千兆以太网可编程交换机，形成了机器人的感知和控制通信中心。这种配置使得Adam能够执行大规模并行动力学模拟和神经网络训练，实现多样化的全身运动控制，真正适用于实际服务场景，适应复杂多变的人类社会环境。灵巧的手部和视觉模块可选配。由于本文关注盲目的运动任务，这些部分未包含在内。Adam（Lite）整体结构的示意图如图2所示，其关节结构及更详细信息见表I。

Motion Capture and Re-Targeting

在我们的研究中，我们探索了各种人体运动数据来源，丰富了我们的训练集，确保我们的模型能够学习到多样化的人体运动特征。最初，我们利用了两个**公共动作捕捉（mocap）数据库**：SFU mocap数据集和CMU mocap数据集。这两个数据集包含了多个动作捕捉序列，涵盖了广泛的人类活动，包括日常动作、体育动作、舞蹈和战斗动作。通过整合这两个数据库，我们为Adam提供了一个多样化且高质量的人体全身运动数据集，这对于训练机器人理解和模仿人体运动模式至关重要。

除了公共数据集之外，我们还使用了高精度的**动作捕捉设备进行定制动作录制**。这种方法使我们能够捕捉特定的运动数据，特别是那些难以在公共数据库中找到的特殊动作或序列，这些动作或序列设计用于特定的实验需求。这些定制的运动数据不仅为我们的数据集增添了多样性，还使我们能够更精确地**微调和优化**我们的模型，使其适应特定的运动任务和挑战。我们没有像humanmimic等方法那样将重定向视为一个优化问题。我们的目标是获得专门为Adam量身定制的高质量运动数据，这导致我们自己手动校准和转换了每个数据集。

