/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"
// 这里的状态量是位姿，包含位置和姿态四元数，因此维数是7，但实际自由度是6（位置3维，姿态3维）
// 其中状态量（位姿变量）的更新。x：优化前的四元数，delta：用旋转矢量表示的增量，x_plus_delta：更新后的四元数
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);        // 优化前的位置
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3); // 优化前的四元数

    Eigen::Map<const Eigen::Vector3d> dp(delta); // 位置增量

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3)); // 将增量由旋转矢量转换为四元数，且增量是个小值

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);        // 更新后的位置
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3); // 更新后的四元数

    p = _p + dp;    // 位置更新采用简单的加法
    q = (_q * dq).normalized(); // 采用四元数乘法进行姿态更新

    return true;
}

// 计算残差关于状态量（位姿变量）的雅克比
// 计算四元数相对于旋转矢量的雅克比矩阵
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
