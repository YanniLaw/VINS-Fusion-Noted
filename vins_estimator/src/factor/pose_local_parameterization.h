/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"
// LocalParameterization类用于对优化参数的维数进行重构，解决非线性优化中的过参数化问题（即待优化参数的实际自由度小于参数本身的维度）。
// 当采用四元数表示姿态时，由于四元数本身的约束（模长为1），其实际的自由度是3而非4；若直接使用四元数进行优化，冗余的维数会带来计算资源的浪费。
// 因此，需要使用LocalParameterization类将四元数重构为3维旋转矢量，在内部优化和更新时实际使用的是3维的等效旋转矢量。
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const; // 在参数正切空间上的更新函数
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };   // 参数的实际维数
    virtual int LocalSize() const { return 6; };    // 正切空间上的参数维数
};
