/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;
// 这个类维护了待边缘化变量块xm与其相关联的变量块xb之间的某单项约束因子Zm（简单理解就是观测信息）
struct ResidualBlockInfo
{   // 形参类似于problem.AddResidualBlock()的参数
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function; // 代价函数
    ceres::LossFunction *loss_function; // 损失函数
    std::vector<double *> parameter_blocks; // 与该单项约束因子相关联的所有的优化变量块
    std::vector<int> drop_set;  // 待边缘化的优化变量块在parameter_blocks中的id

    double **raw_jacobians; // 与该单项约束因子相关的雅克比矩阵
    // Eigen::RowMajor表示行优先，其中用每一行来存储某一个维度的残差关于其优化变量的雅克比
    // 注意，这里它是按照优化变量块来划分子块存储的，所以这里是vector结构
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals; // 与该单项约束因子相关的残差。例如IMU残差是15x1的，视觉残差是2x1的

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 边缘化之后的先验信息类
// 这里的优化变量块分为Xm、Xb；Xm表示待边缘化的优化变量块，Xb表示与Xm有约束关联的优化变量块
class MarginalizationInfo
{
  public:
    MarginalizationInfo(){valid = true;};
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; // Xm与Xb间的所有的约束因子Zm（简单理解就是观测信息）
    int m, n;   // m是需要边缘化的优化变量块Xm的总维度大小localSize。n是保留下来的优化变量块Xb的总维度大小localSize
    // <所有的优化变量块（包括Xm和Xb，即待边缘化的优化变量块以及与它们直接相关联的优化变量块）的内存地址, 优化变量块的维度大小globalSize>
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size; // 边缘化之后保留下来的所有优化变量块的总维度大小globalSize
    // <排序好之后的所有的优化变量块的内存地址, 该变量块在所有优化变量中的起始id，使用localSize>
    std::unordered_map<long, int> parameter_block_idx; //local size
    // <所有的优化变量块的内存地址, 原始优化变量块数据的拷贝的内存指针，使用globalSize>
    // 它是原始优化变量块数据的拷贝，在后面不会参与优化，其值不会变化；用于保存这些优化变量块数据在边缘化时的值
    std::unordered_map<long, double *> parameter_block_data;
    // 下面这三个是边缘化之后保留下来的数据，分别对应了parameter_block_size、parameter_block_idx、parameter_block_data中的second数据
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    // 分别指的是边缘化之后从舒尔补矩阵中恢复出来的等效先验雅克比矩阵和先验残差向量
    // 这里要注意这两个量的维度大小。首先我们假设残差f的实际维度是rx1，雅克比矩阵J的实际维度是rxn；同时，我们知道非线性优化最终要求解的增量方程是HX=b，
    // 其中H=J^T*J，H的维度是nxn；以及b=J^T*f，b的维度nx1；H、b都只和优化变量的维度n有关。因此这两个等效出来的量的维度也只和n有关。
    Eigen::MatrixXd linearized_jacobians; // nxn
    Eigen::VectorXd linearized_residuals; // nx1
    const double eps = 1e-8;
    bool valid;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info; // 上一轮边缘化后保留下来的先验信息
};
