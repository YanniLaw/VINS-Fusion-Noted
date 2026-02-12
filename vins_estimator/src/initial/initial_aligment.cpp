/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "initial_alignment.h"

/**
 * @brief 根据视觉SFM的结果来校正陀螺仪的初始偏置。
 * 由于IMU存在偏置，预积分的旋转与陀螺仪的bias有关，而视觉获得的旋转矩阵不存在bias，所以可以用视觉来标定IMU的旋转bias。
 * 方法是将相邻帧之间通过SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐。因为理论上通过视觉SFM给出的相邻帧间的旋转应等于IMU预积分的旋转值。
 * 局限 / 可能失败的情况:
 * 1. 视觉旋转质量差（特征少、强动态、光照差）会把 bias 校正带偏；
 * 2. 运动激励不足（纯匀速小角速度）时，JTJ可能条件数很差，导致求解不稳定;
 * 3. 默认假设 bias 常值；若 IMU 漂移很大，窗口内统一的 bias 假设不成立，也会导致结果差。
 * @param[in]   all_image_frame：所有图像帧构成的map
 * @param[out]  Bgs：陀螺仪偏置
 * @return      void
*/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i; // frame_i和frame_j分别是all_image_frame中的前、后相邻两帧
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        // 根据视觉SFM恢复出的旋转，将k+1到k图像帧对应的IMU坐标系之间的相对旋转矩阵 R^bk_b(k+1) = (R^cl_bk)^-1 * (R^cl_b(k+1))，转换为四元数
        // 可以看到，这一步需要由视觉恢复出的相机的旋转推出IMU的旋转，因此需要先标定IMU-Camera外参q^b_c
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    //LDLT方法
    delta_bg = A.ldlt().solve(b); // 求解方程 Ax = b
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)  // 在初始化时的偏置为0，所以偏置的增量值就是当前的偏置
        Bgs[i] += delta_bg;
    // 求解出新的陀螺仪偏置后，要对所有的IMU预积分项进行重新传播
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]); // 重传播
    }
}

// 在半径为G的半球找到切面的一对正交基
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized(); // 向量归一化
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    // 施密特正交化。或者从几何的角度来理解：tmp向量减去，tmp向量在a向量上的投影向量，得到的b向量会垂直于a向量
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief   重力向量细化，利用重力向量的模长已知这个先验条件进一步优化g^cl，在其切线空间上用两个变量重新参数化重力。g^=||g^w||*(g^cl)+w1*b1+w2*b2
 *          注意，旋转变换并不会改变向量的模长。
 * 上一步虽然算出了重力向量g，但那个g 是作为一个3D向量自由求解的，算出来的模长（大小）往往不等于标准的9.81 m/s^2（比如可能算出 9.5或10.2）
 * RefineGravity 的核心任务是：在强制重力模长为G（通常是 9.81）的约束下，进一步优化重力的方向，同时微调速度和尺度.
 * 这是一个典型的 流形优化（Manifold Optimization） 问题。
 * 输出模长严格等于 9.81，且方向与 IMU 积分轨迹最吻合的重力向量
 * @param[in]   all_image_frame：所有图像帧构成的map
 * @param[out]  g：重力加速度
 * @param[out]  x：待优化的状态向量：每一IMU速度V[0:n]、2自由度重力参数w:[w1,w2]^T、尺度s。{V^b0_b0, V^b1_b1, ... V^bn_bn, w, s}
 * @return      void
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm(); // g0 = ||g^w||*(g^cl)，大小是惯性系下的重力模长，但方向是计算出的重力向量g^cl
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2); // [b1,b2]
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3); // dg = [w1,w2]^T
            g0 = (g0 + lxly * dg).normalized() * G.norm();  // 模长大小是惯性系下的重力
            //double s = x(n_state - 1);
    }   
    g = g0;
}

/**
 * @brief  初始化出尺度因子s、在cl帧坐标系下表示的重力向量g^cl，以及在body坐标系下表示的每一IMU速度V^bn_bn。
 *          通过将相邻图像帧之间视觉SFM恢复出的位置和速度（也称为预测值），与IMU预积分出来的位置和速度进行对齐，然后求解线性最小二乘，求解出待求变量。
 *    
 * @param[in]   all_image_frame：所有图像帧构成的map
 * @param[out]  g：重力加速度
 * @param[out]  x：待求取的状态向量{V^b0_b0, V^b1_b1, ... V^bn_bn, g^cl, s}
 * @return      void
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;  // 待求取的状态量x的总维度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);    // frame_i和frame_j分别是all_image_frame中的前、后相邻两帧

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;
        // tmp_A(6,10) = H^bk_b(k+1) = [-I*dt           0             (R^bk_cl)*dt*dt/2   (R^bk_cl)*((p^cl_c(k+1))-(p^cl_ck))  ] 
        //                             [ -I    (R^bk_cl)*(R^cl_b(k+1))      (R^bk_cl)*dt                  0                    ]
        // tmp_b(6,1) = z^bk_b(k+1) = [ (α^bk_b(k+1))+(R^bk_cl)*(R^cl_b(k+1))*p^b_c-p^b_c, β^bk_b(k+1)]^T
        // tmp_A * x = tmp_b 求解最小二乘问题
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // 10x10
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // 10x1
        // 这里注意，每循环一次，处理的是相邻两帧frame_i和frame_j，待求解的变量是[v^bk, v^b(k+1), g^cl, s]
        // 而所有待求解的变量是[v^b0, v^b1, v^b2, v^b3, v^b4, v^b5, ... v^bn, g^cl, s]
        // 所以在构建所有帧的完整的A矩阵时，对应项会有一个拼接的操作。这类似于非线性优化里的构造H矩阵和J矩阵
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();   // 对应 v^bk, v^b(k+1)
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b); // ldlt分解求解方程 Ax=b
    double s = x(n_state - 1) / 100.0; // 得到尺度的初始值
    ROS_DEBUG("estimated scale: %f", s); // 例如，这里的s等于0.099555
    g = x.segment<3>(n_state - 4);  // 得到g的初始值g^cl
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose()); // 例如，这里的g为(-0.264097, -0.882286, 9.66412)。模长等于9.70791
    if(fabs(g.norm() - G.norm()) > 0.5 || s < 0)  // 利用先验G判断，如果计算出来的重力加速度与参考值差太大或者尺度为负则说明计算错误
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x); // 重力细化，微调重力向量，再重新优化
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// 使用all_image_frame所有帧间的关系，将视觉SFM结果与IMU预积分结果进行对齐，
// 从而初始化出陀螺仪偏置Bgs、在body坐标系下表示的每一IMU速度V^bn_bn、在cl帧坐标系下表示的重力向量g^cl、以及尺度因子s。
// 理论部分可参考网址:https://blog.csdn.net/iwanderu/article/details/104672579
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);   // 计算陀螺仪的初始偏置

    if(LinearAlignment(all_image_frame, g, x))  // 利用IMU的平移估计重力向量、各bk速度、尺度s
        return true;
    else 
        return false;
}
