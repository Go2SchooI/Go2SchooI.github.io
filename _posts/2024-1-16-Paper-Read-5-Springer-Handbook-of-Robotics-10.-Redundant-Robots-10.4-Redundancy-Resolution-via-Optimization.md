---
layout: post
title:  "Paper Read 5 - Springer Handbook of Robotics 10. Redundant Robots 10.4 Redundancy Resolution via Optimization"
date:   2024-1-16 14:40:00 +0800
tags:
- Manipulator
- Kinematics
toc:  true
math: true

---

In this post, I am going to document my learning process of gradient projection method.

Control schemes for determining the joint trajectories for redundant robots have been developed using both **global and local resolution of redundancy**. The **global redundancy resolution** schemes determine a joint trajectory from the **complete description of the desired end effector trajectory**. The **local redundancy resolution** schemes determine a joint trajectory from the **instantaneous joint motion** required to follow a desired end-effector trajectory. The joint motions are obtained by **satisfying the local optimization of a performance criterion**. 


$$
{\dot{\theta}}=J^{+} {\dot{x}}+\left(I-J^{+} J\right) \dot{\phi}
$$
It provides a joint velocity vector that **minimizes the Euclidean norm** of $$(J {\dot{\theta}}-{\dot{x}})$$ for a given $$\dot{x}$$.

In order to **improve a performance criterion** $$H(\theta)$$ using the gradient projection method, the redundancy is resolved by substituting $$\mathrm{k\nabla H}({\theta})$$ for $$\dot{\phi}$$ and rewriting it as
$$
{\dot{\theta}}=\mathrm{J}^{+} {\dot{\mathrm{x}}}+\mathrm{k}\left(\mathrm{I}-\mathrm{J}^{+} \mathrm{J}\right) \nabla \mathrm{H}({\theta}) .
$$
$$\nabla H({\theta})$$, the gradient vector is described as
$$
\nabla H({\theta})=\left[\partial \mathrm{H} / \partial \theta_1, \partial \mathrm{H} / \partial \theta_2, \ldots \partial \mathrm{H} / \partial \theta_{\mathrm{n}}\right]^{\mathrm{T}}
$$

The scalar constant $$k$$ is **taken to be positive** if $$\mathrm{H}({\theta})$$ is to be **maximized** and **negative** if $$\mathrm{H}({\theta})$$ is to be **minimized**. A larger value of $k$ will optimize at a faster rate but is limited by **bounds on the joint velocities**.

For example, **mechanical joint limits** that are typically present in robot manipulators may be avoided by **minimizing the cost function**
$$
\mathbf{H}(\theta)=\frac{1}{2} \sum_{i=1}^N\left(\frac{\theta_i-\theta_{i, \mathrm{mid}}}{\theta_{i, \mathrm{max}}-\theta_{i, \mathrm{~min}}}\right)^2
$$
where $$\left[\theta_{i, \min }, \theta_{i, \max }\right]$$ is the available range for joint i and $$\theta_{i, \text { mid }}$$ is its midpoint. 

