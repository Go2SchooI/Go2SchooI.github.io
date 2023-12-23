---
layout: post
title:  "Paper Read 4 - A Novel QP-Based Kinematic Redundancy Resolution Method With Joint Constraints Satisfaction"
date:   2023-12-21 17:14:00 +0800
tags:
- Manipulator
toc:  true
math: true


---

## Abstract

1. A novel redundancy resolution method based on a less common **quadratic programming (QP) approach**

2. Velocity-level IK method allows fulfilment of the joint constraints at the **position, velocity, and acceleration levels**.

3. The discretized joint state equations allow the use of **joint accelerations as decision variables** in the QP problem.

## Introduction

Although the **QP formulation of IK** is known—it can be **used to derive the pseudoinverse-based IK** , this paper presents an important enhancement. The scientific novelty of this work is the proposition of a **velocity-level IK method** that allows the fulfilment of the joint **acceleration constraints** together with the **velocity- and position-level constraints**. The elements of the goal function, the Hessian matrix and other necessary quantities, are formulated in the form that uses accelerations instead of the usual velocities.

## Method

#### Classic

The task space variables x, joint space variables q
$$
\mathbf{x}=\mathbf{f}(\mathbf{q})
$$
