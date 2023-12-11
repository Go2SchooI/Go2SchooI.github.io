---
layout: post
title:  "Paper Read 2 - Instantaneous Kinematics"
date:   2023-11-17 19:35:36 +0800
tags:
- Manipulator
toc:  true
math: true
---

Instantaneous kinematics also describes the mapping from joint space to operation space. However, the term "instantaneous" indicates that it does not describe "static" positions, but rather "dynamic" velocities. The function of forward kinematics is shown as follows:
$$
\begin{aligned}
\vec{x} = f(\vec{q})
\end{aligned}
$$

q vector represents the joint position and x vector represents the position and orientation of the end effector.

### Instantaneous Kinematics

When "instantaneous kinematics" solves for the mapping of velocities from joint space to operation space, since velocity describes a change in position over a short period of time, the derivative of position with respect to time, I'm sure it's natural for you to think that we need to solve for such a function:

$$
\frac{d \vec{x}}{d t}=g\left(\frac{d \vec{q}}{d t}\right)
$$

or:

$$
\dot{x}=g(\dot{q})
$$

Our task now is to derive the "instantaneous kinematics" formula from the "forward kinematics" formula:

$$
\frac{d \vec{x}}{d t}=\frac{d f(\vec{q})}{d t}=\frac{d f(\vec{q})}{d \vec{q}} \cdot \frac{d \vec{q}}{d t}
$$

i.e. :

$$
\dot{x}=\frac{d \vec{x}}{d \vec{q}} \cdot \dot{q}
$$

What connects the velocities in joint space to the velocities in operation space is the Jacobi matrix obtained from the derivation of the vectors. Now, let's express this important conclusion mathematically by denoting the derivative of the vector x with respect to the vector q by J:

$$
\dot{x}=J \dot{q}
$$

According to the vector derivation method described at the beginning, J is a matrix. This matrix is not really abstract at all: if we look carefully at each of its elements, we see that its ith row and jth column represent the physical meaning of how the ith translation/rotation direction of the operation space will move when the jth joint moves:

$$
J=\left[\begin{array}{ccc}
\frac{d x_1}{d q_1} & \cdots & \frac{d x_1}{d q_n} \\
\vdots & \ddots & \vdots \\
\frac{d x_m}{d q_1} & \cdots & \frac{d x_m}{d q_n}
\end{array}\right]
$$

For example, the first row and first column indicate that when the first joint moves a certain angle/distance, the end effector correspondingly moves/rotates a certain distance/angle in the direction of x1. If you still think it's too abstract, let's look at an example.

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/931d06cbf13f61cd63e17110b79a7d2f.jpg" alt="931d06cbf13f61cd63e17110b79a7d2f" style="zoom: 33%;" />

Our joint space is (θ1, θ2) and the operation space is (xe, ye), and we have also written out the forward kinematics formulae (making the lengths of the links all 1):

$$
\begin{gathered}
x_e=\cos \left(\theta_1+\theta_2\right)+\cos \theta_1 \\
y_e=\sin \left(\theta_1+\theta_2\right)+\sin \theta_1
\end{gathered}
$$

Then the Jacobi matrix is:

$$
J=\left[\begin{array}{ll}
\frac{d x_e}{d \theta_1} & \frac{d x_e}{d \theta_2} \\
\frac{d y_e}{d \theta_1} & \frac{d y_e}{d \theta_2}
\end{array}\right]=\left[\begin{array}{cc}
-\sin \left(\theta_1+\theta_2\right)-\sin \theta_1 & -\sin \left(\theta_1+\theta_2\right) \\
\cos \left(\theta_1+\theta_2\right)+\cos \theta_1 & \cos \left(\theta_1+\theta_2\right)
\end{array}\right]
$$

Note that in our example, there are two degrees of freedom in joint space and two degrees of freedom in operation space, so our Jacobi matrix is square (square matrix); but a Jacobi matrix is not necessarily square.

Consider the substitution when θ1 = 0 and θ2 = 90°, and we get this Jacobi matrix:

$$
J=\left[\begin{array}{ll}
\frac{d x_e}{d \theta_1} & \frac{d x_e}{d \theta_2} \\
\frac{d y_e}{d \theta_1} & \frac{d y_e}{d \theta_2}
\end{array}\right]=\left[\begin{array}{cc}
-1 & -1 \\
1 & 0
\end{array}\right]
$$

Corresponding to:

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/a2aa00172dbec3bb3bc89be0f1b50564.jpg" alt="a2aa00172dbec3bb3bc89be0f1b50564" style="zoom: 33%;" />

If we keep the first joint motionless and rotate the second joint, then at this one instant the end effector will move only in the x-direction with a velocity of 1 (linear velocity equals the angular velocity multiplied by the radius, i.e. the length of the LINK), and in the y-direction with a velocity of 0, so the second column of the J matrix is [-1, 0].

And if we keep the second joint still and rotate the first joint, the instantaneous velocity of the end effector will be perpendicular to the line connecting the end effector to the axis of the first joint, which has a radius of √2, and the linear velocity will be √2, which breaks down to -1 in the x-direction and 1 in the y-direction, so the first column of the J-matrix is [-1, 1].


$$
\dot{x}=J \dot{q}
$$

This equation shows that the relationship between the velocity of the end effector and the joint velocity is linear! The form of this equation is also exactly the same as the linear equation Ax = b, which we have used various methods (Gaussian elimination being one of them) to solve in linear algebra: if we want the end effector to move at a certain velocity, and we find the corresponding joint velocity, the problem is a problem of solving a linear equation!



We call the Jacobi matrix obtained by describing the linear and angular velocities in **Cartesian coordinates** and the end effector's velocity in the **base frame** of the manipulator as the reference system the basic Jacobi matrix; all other representations (e.g., changing Cartesian coordinates to cylindrical coordinates, spherical coordinates, changing angles to Euler angles or quaternion quaternions, etc.) can be converted from this basic Jacobi matrix. According to the definition of the basic Jacobi matrix above, the speed of the end effector can be written as follows:

$$
\dot{x}=\left[\begin{array}{c}
v_x \\
v_y \\
v_z \\
\omega_x \\
\omega_y \\
\omega_z
\end{array}\right]=\left[\begin{array}{c}
\vec{v} \\
\vec{\omega}
\end{array}\right]
$$

Correspondingly, the Jacobi matrix can be written as:

$$
J=\left[\begin{array}{l}
J_v \\
J_\omega
\end{array}\right]
$$

The upper part corresponds to the **linear velocity** and the lower part to the **angular velocity**.

## Linear velocity component (Jv)

If our robotic arm is a bit more complex and we need to use chi-square coordinate transformations to find the positive kinematics formula, how should the Jacobi matrix be solved?

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/252c0c579bc9f6c131a2b9a3b90060f5.jpg" alt="252c0c579bc9f6c131a2b9a3b90060f5" style="zoom:33%;" />

This is the positive kinematics expression we derived for the end effector:

$$
{ }_e^0 T=\left[\begin{array}{cccc}
\times & \times & \times & l_2 c \theta_1 c \theta_3+l_1 c \theta_1 \\
\times & \times & \times & l_2 s \theta_1 c \theta_3+l_1 s \theta_1 \\
\times & \times & \times & l_2 s \theta_3+d_2 \\
0 & 0 & 0 & 1
\end{array}\right]
$$

The only thing we need is the part of the figure circled by the red box, and if you are good at coordinate transformations, you should immediately see that it represents the position of the end effector w.r.t frame{0}. So, for Jv, we just need to take this 3×1 vector (xe, ye, ze) circled in the red frame and solve for the joint space vectors (θ1, d2, θ3, θ4)! Following the rules for vector derivation, we will get a 3×4 matrix as follows:

$$
J_v=\left[\begin{array}{cccc}
-l_2 s \theta_1 c \theta_3-l_1 s \theta_1 & 0 & -l_2 c \theta_1 s \theta_3 & 0 \\
l_2 c \theta_1 c \theta_3+l_1 c \theta_1 & 0 & -l_2 s \theta_1 s \theta_3 & 0 \\
0 & 1 & l_2 c \theta_3 & 0
\end{array}\right]
$$

## Angular velocity component (Jw)

Let's start by looking at the simplest planar robotic arm:

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/a168c50514415dccc4bc3133e4759bb0.jpg" alt="a168c50514415dccc4bc3133e4759bb0" style="zoom:33%;" />

Of course, this time we are only concerned with the orientation of the end effector. For a planar robotic arm, the end effector has only one degree of rotational freedom α, which is labelled in the figure (set to 0° when coinciding with the x-axis, and turning from the x-axis to the y-axis is a positive direction). At this point, our operation space is (α), and the joint space is still (θ1,θ2); by definition, we require the angular velocity part of the Jacobi matrix as follows:

$$
J_\omega=\left[\begin{array}{ll}
\frac{d \alpha}{d \theta_1} & \frac{d \alpha}{d \theta_2}
\end{array}\right]
$$

For this planar robotic arm, it is easy to see that: α will turn by as many angles as θ1 turns; ditto for θ2 - so Jw should be [1, 1].
What's the point of this example? Hopefully it helps you to establish an intuitive impression and basic idea - **how many angles a rotary joint of a robot arm turns around a certain axis, and how many angles its end effector turns around that axis accordingly**; in the case of a planar robot arm, this means that the rotational speed of a rotary joint is multiplied by 1 to get the end effector that it causes ( Contribute) the speed of rotation of the end effector, so Jw above is [1, 1].


$$
J_\omega=\left[\begin{array}{llll}
{ }^0 \hat{z}_1 & \overrightarrow{0} & { }^0 \hat{z}_3 & { }^0 \hat{z}_4
\end{array}\right]
$$


<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/%E6%8E%A8%E5%AF%BC_13_1702283296643.png" alt="推导_13_1702283296643" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/Go2SchooI/blogImg@main/img/%E6%8E%A8%E5%AF%BC_14_1702283311374.png" alt="推导_14_1702283311374" style="zoom: 25%;" />