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
 * 所以最终就是 找一个陀螺仪 bias 修正量 𝛿bg(真实的零偏就是当前的估计值bg加上一个微小的修正量 𝛿bg)，
 * 让IMU预积分出来的相对旋转，尽量和视觉估计出来的相对旋转一致。
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
    // 真正的 IMU 旋转，应该是用真实的零偏去积分得到的;而真实的零偏就是当前的估计值bg加上一个微小的修正量 𝛿bg
    // 根据李群上的一阶泰勒展开近似，加上了修正量后的 IMU 预积分旋转，可以表示为“原本的旋转”乘上一个“微小的旋转扰动”: 
    // 即q^pre_ij(bg+𝛿bg) ≈ q^pre_ij(bg) * Exp(J_γ_bg * 𝛿bg)，其中Exp()是李群上的指数映射，J_γ_bg是预积分旋转增量对陀螺仪bias的雅可比矩阵
    // 我们希望修正后的 IMU 预积分旋转增量等于视觉测量的旋转，所以构建误差方程:
    // q^pre_ij(bg) * Exp(J_γ_bg * 𝛿bg) = q^sfm_ij
    // 移项得到: Exp(J_γ_bg * 𝛿bg) = q^pre_ij(bg)^-1 * q^sfm_ij (右边这一串就是误差四元数q_error)
    // 又根据四元数和微小旋转角的关系，如果一个旋转角度 θ 非常小，它的四元数可以近似写为 q ≈ [1, 1/2 * θ]^T
    // 由于等式左边是微小旋转 Exp(J_γ_bg * 𝛿bg)，所以有[1, 1/2 * J_γ_bg * 𝛿bg] = q_error
    // 都提取虚部出来，1/2 * J_γ_bg * 𝛿bg = vec(q_error)
    // 最后就得到了一个线性方程组 J_γ_bg * 𝛿bg = 2 * vec(q_error)，于是就构成一个线性方程 Ax = b 来求解 𝛿bg就可以了
    // 但是由于相邻的图像帧有很多对，把滑动窗口里所有帧的方程全部叠加在一起，这就变成了一个典型的超定方程组(方程个数大于未知数个数)，
    // 所以为了求解这个超定方程组，我们需要构建正规方程 A^T * A * 𝛿bg = A^T * b 来求解 𝛿bg
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        // 根据视觉SFM恢复出的旋转，将k+1到k图像帧对应的IMU坐标系之间的相对旋转矩阵 R^bk_b(k+1) = (R^cl_bk)^-1 * (R^cl_b(k+1))，转换为四元数
        // 可以看到，这一步需要由视觉恢复出的相机的旋转推出IMU的旋转，因此需要先标定IMU-Camera外参q^b_c
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R); // 视觉sfm求出来的旋转增量四元数 q^sfm_ij
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG); // 预积分旋转增量对陀螺仪bias的雅可比矩阵
        // frame_j->second.pre_integration->delta_q.inverse() * q_ij 就是预积分旋转增量与视觉旋转增量之间的误差四元数q_error
        // 即q_erroror = q_pre^T * q_ij, 如果误差很小，那么 𝛿q ≈ [1, 1/2 * 𝛿θ]^T
        // 取vec()就是误差四元数的向量部分，乘以2就是𝛿θ了(小角度近似下，误差四元数的向量部分约等于旋转误差的一半)
        // 所以这里tmp_b其实是在在计算一个3 维小角度旋转误差
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    //LDLT方法
    delta_bg = A.ldlt().solve(b); // 求解方程 J^T*T*x = J^T*b
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)  // 在初始化时的偏置为0，所以偏置的增量值就是当前的偏置
        Bgs[i] += delta_bg; // 假设整个初始化窗口内的陀螺仪零偏是同一个常数，所以下面用Bgs[0]来重新传播
    // 求解出新的陀螺仪偏置后，要对所有的IMU预积分项进行重新传播
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]); // 重传播
    }
}

/**
 * @brief 为三维空间中的重力向量g0，寻找它所在球面上的切平面，并求出这个切平面的两个正交基底向量
 * 
 * @param g0 重力向量
 * @return MatrixXd 
 */
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized(); // 归一化(后面只关心方向，不关心模长)，确定球面法向量，我们求的切平面必须完全垂直于这个方向量a
    Vector3d tmp(0, 0, 1);
    // 奇异性处理：如果a向量和tmp向量平行了，那么它们的叉乘就会得到零向量，这时就无法构造出切平面了，所以需要换一个tmp向量来构造切平面
    // 为了在切平面上找一个向量，最简单的办法是随便拿一个不与 a 共线的参考向量，
    // 然后把它“拍”到切平面上。通常我们默认选 Z轴方向的单位向量 (0, 0, 1) 作为参考向量 tmp
    if(a == tmp) // 也可以用这个|a*tmp| >1-eps，说明a和tmp平行了，这时换一个向量来构造切平面
        tmp << 1, 0, 0;
    // 施密特正交化。或者从几何的角度来理解：tmp向量减去，tmp向量在a向量上的投影向量，得到的b向量(norm为单位向量)会垂直于a向量
    // a.transpose() * tmp 这个向量内积是tmp在a上的投影长度，乘以a就是tmp在a上的投影向量(也就是tmp在法向量a方向量的平行分量)，
    // tmp - 这个投影向量 ， 剩下的就是垂直分量了，就是b向量，最后再归一化就得到了单位向量
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b); // 叉乘构造第二个正交向量c也垂直于a向量，并且与b向量也垂直，所以a、b、c三者两两正交，构成一个右手坐标系
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief   重力向量细化，利用重力向量的模长已知这个先验条件进一步优化g^cl，在其切线空间上用两个变量重新参数化重力。
 *  g^=||g^w||*(g^cl)+w1*b1+w2*b2 注意，旋转变换并不会改变向量的模长。
 * 上一步虽然算出了重力向量g，但那个g 是作为一个3D向量自由求解的(无约束)，算出来的模长（大小）往往不等于标准的9.81 m/s^2（比如可能算出 9.5或10.2）
 * 而我们的重力向量满足一个先验条件：它的模长应该等于标准重力加速度的模长（约9.81 m/s^2）。
 * 因此，重力向量实际上被约束在一个半径为 9.81 的球面上。它只有 2 个自由度（重力向量的方向）
 * 如果直接拿 3 维g 去优化，会有两个问题:
 * 1. 可能解出来的模长不精确; 2. 会引入不必要的自由度，影响稳定性.
 * 用这种方法来优化重力向量，既满足了模长约束，又避免了不必要的自由度，提高了优化的稳定性和精度，同时和尺度和速度的耦合更加合理。
 * 所以我们需要对重力向量进行细化，使得它既满足与IMU预积分轨迹的对齐，又满足模长为9.81的约束。
 * RefineGravity 的核心任务是：在强制重力模长为G（通常是 9.81）的约束下(把重力强制约束回这个球面上)，
 * 并利用切线空间（Tangent Space）只去优化它的 2 个方向自由度,
 * 进一步优化重力的方向，同时微调速度和尺度，提高整个系统的初始对齐精度。
 * 这是一个典型的 流形优化（Manifold Optimization） 问题。
 * 输出模长严格等于 9.81，且方向与 IMU 积分轨迹最吻合的重力向量
 * @param[in]   all_image_frame：所有图像帧构成的map
 * @param[out]  g：重力加速度
 * @param[out]  x：待优化的状态向量：每一IMU速度V[0:n]、2自由度重力参数w:[w1,w2]^T、尺度s。{V^b0_b0, V^b1_b1, ... V^bn_bn, w, s}
 * @return      void
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm(); // g0 = g^cl / ||g^cl||* ||g^w||，大小是惯性系下的重力模长，但方向是计算出的重力向量g^cl
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1; // 因为现在待求解的重力向量变成了两个
    // 其他步骤与之前的线性对齐基本一样，区别在于现在构建的A矩阵和b向量中，重力向量g^cl被重新参数化成了两个变量w1和w2，
    // 这两个变量分别是重力向量在切线空间上两个正交基b1和b2上的投影系数(其实就是扰动)
    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++) // 执行4次迭代，每次迭代都在当前的重力向量g0的切线空间上优化w1和w2，更新重力向量g0，逐步逼近最优解
    {
        MatrixXd lxly(3, 2); // [b1,b2]
        // 将重力向量参数化为 g = g0 + 𝛿g,  𝛿g = w1*b1 + w2*b2，其中b1和b2是g0所在球面上的切平面的两个正交基底向量，
        // w1和w2是重力向量在切平面上两个基底向量的投影系数(也就是扰动)，通过优化w1和w2来优化重力向量的方向
        lxly = TangentBasis(g0); 
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9); // 相比之前少了一列，因为现在是重力扰动w1和w2两个变量，而不是重力向量g^cl的三个分量了
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0; // 因为重力方向变化会影响尺度估计，两者是耦合的    
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
            // 更新重力向量g0 = (g0 + 𝛿g) / || (g0 + 𝛿g) || * ||g^w||, 𝛿g = w1*b1 + w2*b2
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
        // 每对相邻帧贡献 6 个标量约束(位置预积分约束3维，速度预积分约束3维)，
        // 10个待求变量（每帧的速度3个分量，共6个；重力向量g^cl的3个分量；尺度s的1个分量）
        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;
        // tmp_A(6,10) = H^bk_b(k+1) = [-I*dt           0             (R^bk_cl)*dt*dt/2   (R^bk_cl)*((p^cl_c(k+1))-(p^cl_ck))  ] 
        //                             [ -I    (R^bk_cl)*(R^cl_b(k+1))      (R^bk_cl)*dt                  0                    ]
        // tmp_b(6,1) = z^bk_b(k+1) = [ (α^bk_b(k+1))+(R^bk_cl)*(R^cl_b(k+1))*p^b_c-p^b_c, β^bk_b(k+1)]^T
        // tmp_A * x = tmp_b 求解最小二乘问题

        // 为什么单独对 s 对应的列 / 100.0？
        // 在我们的未知数向量 x = [v_i, v_j, g, s]^T 中，各个变量的物理量级差异非常大
        // 速度v大概是1到2的量级， 重力g是 9.8 左右， 但是尺度 s 有可能极其微小（比如 0.01 甚至更小，取决于视觉特征点的三角化深度）
        // 而在方程左侧（即 tmp_A 矩阵里），跟 s相乘的系数是(T_j - T_i)，这个平移增量在纯视觉中通常被归一化得比较大（比如在 1 到 10 之间）
        // 数值灾难来了：如果一个矩阵里，有的列数值很大（比如 10），有的列数值很小（比如 dt^2大概是 0.001），当你用 LDLT 去求逆解方程组时，
        // 计算机的浮点数精度会被严重吃掉，导致算出来的结果完全是错的（矩阵趋于奇异）。
        // 解法：变量代换（Pre-conditioning）
        // 作者人为地把矩阵中 s所在的这一列缩小了 100 倍（/ 100.0），数学上，这相当于我们不再直接求解s，
        // 而是去求解一个叫 s'的中间变量，且满足 s' = 100*s
        // 因为我们把方程里的系数缩小了 100 倍，为了保持等式成立，解出来的变量自然就放大了 100 倍
        // 这样s' 的数值量级就被“拉拔”到了和速度 v、重力 g 差不多的水平（比如从 0.01 变成了 1.0），矩阵运算就稳定了
        // 所以最后求解完，必须再把结果除以 100 还原
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0; // 构造时：系数列除以100，相当于求解 s' = 100*s
     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();

        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;
        // 权重矩阵，理论上应该是预积分的协方差矩阵的逆，但为了简化，这里暂时设为单位矩阵
        // 所以目前是等权重地使用位置和速度约束，没有真的利用 IMU 协方差去加权
        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // 10x10
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // 10x1
        // 这里注意，每循环一次，处理的是相邻两帧frame_i和frame_j，待求解的变量是[v^bk, v^b(k+1), g^cl, s]
        // 而所有待求解的变量是[v^b0, v^b1, v^b2, v^b3, v^b4, v^b5, ... v^bn, g^cl, s]
        // 所以在构建所有帧的完整的A矩阵时，对应项会有一个拼接的操作。这类似于非线性优化里的构造H矩阵和J矩阵
        // 可以用下图理解A的矩阵结构: 
        //         v0  v1  v2  v3  ... vn  g   s
        // v0 [ ██  ██              |  ██  █ ]
        // v1 [ ██  ██  ██          |  ██  █ ]
        // v2 [     ██  ██  ██      |  ██  █ ]
        // v3 [         ██  ██      |  ██  █ ]
        // ...                 ...  |        
        // g  [ ██  ██  ██  ██  ... |  ██  █ ]
        // s  [ █   █   █   █   ... |  █   █ ]
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();   // 对应 v^bk, v^b(k+1)
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>(); // 对应 g^cl, s
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>(); // 对应 v^bk, v^b(k+1) 与 g^cl, s 之间的交叉项
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 这是一个整体的缩放，在相机帧率很高时（比如 30Hz），相邻两帧的时间差 dt 只有约 0.033 秒
    // 矩阵 A 里面有很多项是包含 dt 甚至 dt^2 的，这些数字非常非常小（0.001 级别）
    // 经过 A.transpose() * A 矩阵乘法后，有些元素可能会变成 10^{-6} 级别
    // 如果直接拿这种极小数值的矩阵去交给 Eigen 库求解 ldlt().solve()，很容易因为数值下溢（Underflow）而导致解算失败
    // 两边同时乘以 1000（相当于 1000A x = 1000b），解 x 是完全不变的，但这把内存里的浮点数整体放大了，避开了精度丢失的陷阱
    A = A * 1000.0; // 乘以1000是为了数值稳定，不改变解，防止A矩阵的元素过小导致求解不稳定
    b = b * 1000.0;
    x = A.ldlt().solve(b); // ldlt分解求解方程 Ax=b
    // 前面是对尺度变量本身的重参数化
    // 由于在构建A矩阵时，尺度s的系数列被除以了100，所以这里求解出来的x中尺度s的值是被放大了100倍的，所以要除以100还原回来
    // 如果不进行重参数化，直接求解s的话，由于s的量级通常比较小（比如0.1），而其他变量的量级可能比较大（比如速度可能是几米每秒），
    // 这样A矩阵中尺度s相关的元素就会很小，直接求解小量容易导致矩阵病态，导致数值不稳定，求解出来的s可能会有很大的误差。   
    double s = x(n_state - 1) / 100.0; // 得到尺度的初始值， // 提取时：再除以100，从 s' 还原真正的 s
    ROS_DEBUG("estimated scale: %f", s); // 例如，这里的s等于0.099555
    g = x.segment<3>(n_state - 4);  // 得到g的初始值g^cl
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose()); // 例如，这里的g为(-0.264097, -0.882286, 9.66412)。模长等于9.70791
    if(fabs(g.norm() - G.norm()) > 0.5 || s < 0)  // 利用先验G判断，如果计算出来的重力加速度与参考值差太大或者尺度为负则说明计算错误
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x); // 重力细化，微调重力向量，再重新优化(现在的x，里面的重力向量部分是两维了)
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s; // 回带更新
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
