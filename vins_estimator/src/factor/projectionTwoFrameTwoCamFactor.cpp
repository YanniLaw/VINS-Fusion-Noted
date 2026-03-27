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

#include "projectionTwoFrameTwoCamFactor.h"

Eigen::Matrix2d ProjectionTwoFrameTwoCamFactor::sqrt_info; // 重投影误差的平方根信息矩阵
double ProjectionTwoFrameTwoCamFactor::sum_t;

/**
 * @brief Construct a new Projection Two Frame Two Cam Factor:: Projection Two Frame Two Cam Factor object
 * 
 * @param _pts_i  该路标点在start_frame帧下的归一化相机坐标
 * @param _pts_j  该路标点在当前帧下的归一化相机坐标
 * @param _velocity_i  该路标点在start_frame帧相机归一化平面上的速度
 * @param _velocity_j  该路标点在当前帧相机归一化平面上的速度
 * @param _td_i  start_frame帧的imu-camera的同步时钟偏差
 * @param _td_j  当前帧的imu-camera的同步时钟偏差
 */
ProjectionTwoFrameTwoCamFactor::ProjectionTwoFrameTwoCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                                                               const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                                               const double _td_i, const double _td_j) : 
                                                               pts_i(_pts_i), pts_j(_pts_j), 
                                                               td_i(_td_i), td_j(_td_j)
{
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;

#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

// parameters[0~4]分别对应了优化变量块：para_Pose[i], para_Pose[j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]
// 该函数返回计算出的残差residuals，以及雅克比矩阵jacobians
bool ProjectionTwoFrameTwoCamFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Eigen::Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    double inv_dep_i = parameters[4][0]; // 路标点的逆深度

    double td = parameters[5][0];   // 相机-IMU的时钟偏差

    // 在VIO系统中包含视觉传感器（相机）和惯性传感器(IMU)，系统对这两个传感器分别进行采样，
    // 获得相应的数据（图像、IMU数据）与对应的采样时间戳（记录传感器测量值的瞬时时间），
    // 通常，我们假设记录的采样时间戳就是传感器真实的采样时间点，比如相机曝光时刻（通常曝光持续几毫秒到几十毫秒，我们认为曝光时刻为曝光时续的中间时刻）
    // 然而，由于硬件系统存在触发延时、曝光时间、数据传输延迟以及没有准确的同步时钟等问题，
    // 物理世界中相同时刻的IMU、相机帧数据，其记录的时间戳与真实采样时间存在一个td的时间偏差。
    // 在这里假设IMU的时间戳是准确的，图像数据的时间戳可能是有偏差的，所以这里的时间偏差就指的是IMU与相机数据之间的相对时间戳偏差。
    // 这里假设时间偏差是一个未知的常数，并假设图像上特征点像素在短时间内以恒定速度在图像平面上移动（每个像素各自的运动是匀速的），
    // 这就可以估计出在对应时间戳的特征点像素的位置，
    // 先计算出每个特征点像素的移动速度，然后根据速度值提前补偿时间戳不对齐对图像特征点的位置影响；
    // 这样就将相机与IMU的时间偏移td，转换为在图像平面上特征点位置的延迟。

    // 在估计得到td之后，会利用该td对图像数据的时间戳进行补偿修正。
    // pts_i/ pts_j 表示在真实采样时刻对应的特征位置，因为在前面已补偿过了图像数据的时间戳，
    // 所以在这里，迭代优化的是新的时间间隔δtd，直至最后将δtd收敛到0。
    // 随着δtd的减小，我们的假设（特征点在短时间隔内以恒定速度在图像平面上移动）越来越合理，
    // 即使在开始时存在巨大的时间偏移（如数百毫秒），该过程也将逐渐地从粗略到精细地补偿它。 
    // velocity_i是该角点在归一化平面的运动速度。
    // 所以最后得到的pts_i_td是处理时间同步误差，角点在归一化平面的坐标。
    // 这两行代码表示了对相机-IMU时间戳偏移的支持
    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;
    // l路标点在ci帧的相机3D坐标P^ci_l
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    // R^b_c * P^ci_l + p^b_c
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    // R^w_bi * ( R^b_c * P^ci_l + p^b_c ) + p^w_bi
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    // R^bj_w * ( R^w_bi * ( R^b_c * P^ci_l + p^b_c ) + p^w_bi - p^w_bj )
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    // 计算出来的l路标点在cj帧的相机3D坐标P^cj_l = R^c_b * ( R^bj_w * ( R^w_bi * ( R^b_c * P^ci_l + p^b_c ) + p^w_bi - p^w_bj ) - p^b_c )
    Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2); 
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    // 这里假设相机3D坐标pts_camera_j为(x,y,z)，像素观测量的反投影的相机归一化坐标pts_j_td为(u,v,1)
#ifdef UNIT_SPHERE_ERROR 
    // 实际上，真实的视觉向量差rc = w1 * b1 + w2 * b2 = (b1,b2) * (w1,w2)^T。其中w1、w2是标量系数，r、b1、b2是3维列向量。
    // 这里的残差residual，其实是构成切平面的两个正交基的坐标(w1,w2)^T = (b1,b2)^T * rc。
    // 为方便表示，这里将pts_camera_j的模长sqrt(x^2+y^2+z^2)记为norm，pts_j_td的模长sqrt(u^2+v^2+1)记为piexl_norm。
    // 则残差residual表示为2维向量：tangent_base * (x/norm - u/piexl_norm, y/norm - v/piexl_norm, z/norm - 1/piexl_norm)^T
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>(); // 残差residual表示为2维向量(x/z - u, y/z - v)^T
#endif

    residual = sqrt_info * residual;
    // 补偿了时间偏差后的视觉重投影残差的雅克比矩阵。优化变量块是两图像帧的[p^w_bi、q^w_bi]，[p^w_bj、q^w_bj]，[p^b_c、q^b_c]，[λl]，[td]
    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3); // 2x3，表示残差residual对pts_camera_j的导数
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm(); // 模长sqrt(x^2+y^2+z^2)
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
        // 残差r关于pts_camera_j(x,y,z)的导数为：tangent_base * 
        // [ ((x/norm - u/piexl_norm) / x)`, ((x/norm - u/piexl_norm) / y)`, ((x/norm - u/piexl_norm) / z)` ] = [ 1/norm - x^2/norm^3, -x*y/norm^3, -x*z/norm^3 ]
        // [ ((y/norm - v/piexl_norm) / x)`, ((y/norm - v/piexl_norm) / y)`, ((y/norm - v/piexl_norm) / z)` ] = [ -y*x/norm^3, 1/norm - y^2/norm^3, -y*z/norm^3 ]
        // [ ((z/norm - 1/piexl_norm) / x)`, ((z/norm - 1/piexl_norm) / y)`, ((z/norm - 1/piexl_norm) / z)` ] = [ -z*x/norm^3, -z*y/norm^3, 1/norm - z^2/norm^3 ]
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
        // 残差r关于pts_camera_j(x,y,z)的导数为：[ ((x/z - u) / x)`, ((x/z - u) / y)`, ((x/z - u) / z)` ] = [ 1/z, 0, -x^2/z ]
        //                                    [ ((y/z - v) / x)`, ((y/z - v) / y)`, ((y/z - v) / z)` ] = [ 0, 1/z, -y^2/z ]
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric2.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric2.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric2.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri; 
            jaco_ex.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * ric * -Utility::skewSymmetric(pts_camera_i);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(jacobians[3]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = - ric2.transpose();
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        if (jacobians[4])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[4]);
#if 1
            jacobian_feature = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
#else
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
        }
        if (jacobians[5])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[5]);
            jacobian_td = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                          sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

void ProjectionTwoFrameTwoCamFactor::check(double **parameters)
{
    double *res = new double[15];
    double **jaco = new double *[6];
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 7];
    jaco[4] = new double[2 * 1];
    jaco[5] = new double[2 * 1];
    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[3]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[4]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[5]) << std::endl
              << std::endl;

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Eigen::Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    double inv_dep_i = parameters[4][0];

    double td = parameters[5][0];
    //pts_i_td 处理时间同步误差时间后，角点在归一化平面的坐标。
    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;

    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);


    Eigen::Vector2d residual;
#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 26> num_jacobian;
    for (int k = 0; k < 26; k++)
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

        double inv_dep_i = parameters[4][0];

        double td = parameters[5][0];

        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            Pi += delta;
        else if (a == 1)
            Qi = Qi * Utility::deltaQ(delta);
        else if (a == 2)
            Pj += delta;
        else if (a == 3)
            Qj = Qj * Utility::deltaQ(delta);
        else if (a == 4)
            tic += delta;
        else if (a == 5)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 6)
            tic2 += delta;
        else if (a == 7)
            qic2 = qic2 * Utility::deltaQ(delta);
        else if (a == 8)
        {
            if(b == 0)
                inv_dep_i += delta.x();
            else
                td += delta.y();
        }

        Eigen::Vector3d pts_i_td, pts_j_td;
        pts_i_td = pts_i - (td - td_i) * velocity_i;
        pts_j_td = pts_j - (td - td_j) * velocity_j;

        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);

        Eigen::Vector2d tmp_residual;
#ifdef UNIT_SPHERE_ERROR 
        tmp_residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
        tmp_residual = sqrt_info * tmp_residual;
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian.block<2, 6>(0, 0) << std::endl;
    std::cout << num_jacobian.block<2, 6>(0, 6) << std::endl;
    std::cout << num_jacobian.block<2, 6>(0, 12) << std::endl;
    std::cout << num_jacobian.block<2, 6>(0, 18) << std::endl;
    std::cout << num_jacobian.block<2, 1>(0, 24) << std::endl;
    std::cout << num_jacobian.block<2, 1>(0, 25) << std::endl;
}
