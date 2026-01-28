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

#include "solve_5pts.h"


namespace cv {
    void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
    {

        Mat E = _E.getMat().reshape(1, 3);
        CV_Assert(E.cols == 3 && E.rows == 3);

        Mat D, U, Vt;
        SVD::compute(E, D, U, Vt);

        if (determinant(U) < 0) U *= -1.;
        if (determinant(Vt) < 0) Vt *= -1.;

        Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        W.convertTo(W, E.type());

        Mat R1, R2, t;
        R1 = U * W * Vt;
        R2 = U * W.t() * Vt;
        t = U.col(2) * 1.0;

        R1.copyTo(_R1);
        R2.copyTo(_R2);
        t.copyTo(_t);
    }

    int recoverPose( InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                         OutputArray _R, OutputArray _t, InputOutputArray _mask)
    {

        Mat points1, points2, cameraMatrix;
        _points1.getMat().convertTo(points1, CV_64F);
        _points2.getMat().convertTo(points2, CV_64F);
        _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

        int npoints = points1.checkVector(2);
        CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                                  points1.type() == points2.type());

        CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

        if (points1.channels() > 1)
        {
            points1 = points1.reshape(1, npoints);
            points2 = points2.reshape(1, npoints);
        }

        double fx = cameraMatrix.at<double>(0,0);
        double fy = cameraMatrix.at<double>(1,1);
        double cx = cameraMatrix.at<double>(0,2);
        double cy = cameraMatrix.at<double>(1,2);

        points1.col(0) = (points1.col(0) - cx) / fx;
        points2.col(0) = (points2.col(0) - cx) / fx;
        points1.col(1) = (points1.col(1) - cy) / fy;
        points2.col(1) = (points2.col(1) - cy) / fy;

        points1 = points1.t();
        points2 = points2.t();

        Mat R1, R2, t;
        decomposeEssentialMat(E, R1, R2, t);
        Mat P0 = Mat::eye(3, 4, R1.type());
        Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
        P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
        P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
        P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
        P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

        // Do the cheirality check.
        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between postive and negtive.
        double dist = 50.0;
        Mat Q;
        triangulatePoints(P0, P1, points1, points2, Q);
        Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask1 = (Q.row(2) < dist) & mask1;
        Q = P1 * Q;
        mask1 = (Q.row(2) > 0) & mask1;
        mask1 = (Q.row(2) < dist) & mask1;

        triangulatePoints(P0, P2, points1, points2, Q);
        Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask2 = (Q.row(2) < dist) & mask2;
        Q = P2 * Q;
        mask2 = (Q.row(2) > 0) & mask2;
        mask2 = (Q.row(2) < dist) & mask2;

        triangulatePoints(P0, P3, points1, points2, Q);
        Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask3 = (Q.row(2) < dist) & mask3;
        Q = P3 * Q;
        mask3 = (Q.row(2) > 0) & mask3;
        mask3 = (Q.row(2) < dist) & mask3;

        triangulatePoints(P0, P4, points1, points2, Q);
        Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask4 = (Q.row(2) < dist) & mask4;
        Q = P4 * Q;
        mask4 = (Q.row(2) > 0) & mask4;
        mask4 = (Q.row(2) < dist) & mask4;

        mask1 = mask1.t();
        mask2 = mask2.t();
        mask3 = mask3.t();
        mask4 = mask4.t();

        // If _mask is given, then use it to filter outliers.
        if (!_mask.empty())
        {
            Mat mask = _mask.getMat();
            CV_Assert(mask.size() == mask1.size());
            bitwise_and(mask, mask1, mask1);
            bitwise_and(mask, mask2, mask2);
            bitwise_and(mask, mask3, mask3);
            bitwise_and(mask, mask4, mask4);
        }
        if (_mask.empty() && _mask.needed())
        {
            _mask.create(mask1.size(), CV_8U);
        }

        CV_Assert(_R.needed() && _t.needed());
        _R.create(3, 3, R1.type());
        _t.create(3, 1, t.type());

        int good1 = countNonZero(mask1);
        int good2 = countNonZero(mask2);
        int good3 = countNonZero(mask3);
        int good4 = countNonZero(mask4);

        if (good1 >= good2 && good1 >= good3 && good1 >= good4)
        {
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask1.copyTo(_mask);
            return good1;
        }
        else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
        {
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask2.copyTo(_mask);
            return good2;
        }
        else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
        {
            t = -t;
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask3.copyTo(_mask);
            return good3;
        }
        else
        {
            t = -t;
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask4.copyTo(_mask);
            return good4;
        }
    }

    int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                         OutputArray _t, double focal, Point2d pp, InputOutputArray _mask)
    {
        Mat cameraMatrix = (Mat_<double>(3,3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
        return cv::recoverPose(E, _points1, _points2, cameraMatrix, _R, _t, _mask);
    }
}

/**
 * @brief 已知两组匹配好的特征点，利用对极几何（Epipolar Geometry）原理，算出这两个相机位置之间的相对旋转（R）和平移（T）
 * 
 * @param corres 匹配好的特征点对，分别在两帧下的相机归一化平面坐标
 * @param Rotation     当前帧到参考帧的旋转矩阵
 * @param Translation  当前帧到参考帧的平移向量(注意: 这里只有方向可靠，尺度未知)
 * @return true 
 * @return false 
 */
bool MotionEstimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    // 理论上本质矩阵/基础矩阵最少 5/7/8 点即可（算法不同），但这里设 15 是为了给 RANSAC 足够的采样与鲁棒性冗余。初始化阶段点对质量不稳定（外点多、视差不足），门槛略高很常见
    if (corres.size() >= 15)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        // 虽然函数名叫 findFundamentalMat，但因为输入的是归一化坐标，所以计算出来的实际上是 本质矩阵 (Essential Matrix, E)
        /**
         *  Mat cv::findFundamentalMat(  通过RANSAC算法求解两幅图像之间的本质矩阵E
         *      InputArray  points1,            第一幅图像点的数组
         *      InputArray  points2,            第二幅图像点的数组
         *      int     method = FM_RANSAC,     计算基本矩阵的方法。FM_7POINT：7点法，最少七对点。FM_8POINT：8点法，最少八对点。
         *                                      FM_RANSAC：RANSAC 算法，RANSAC 估计 + 外点剔除(工程中最常用)。FM_LMEDS：LMedS 算法，不需要指定阈值。
         *      double ransacReprojThreshold = 3.,  这个参数只用于方法RANSAC 或 LMedS；它是点到对极线的最大距离，超过这个值的点(外点)将被舍弃，不用于后面的计算
         *                                          单位与输入点坐标一致: 1. 如果你传的是像素坐标，阈值单位是像素; 2. 如果你传的是归一化坐标,阈值也是归一化坐标单位(无量纲)
         *      double  confidence = 0.99,          这个参数只用于方法RANSAC 或 LMedS；它表示矩阵正确的可信度
         *      OutputArray mask = noArray()    与InputArray维数相同的输出掩码，标记内点(匹配正确,值为1)及外点(匹配错误，值为0)
         *                                      mask后续怎么用? 通常根据 mask 把输入的特征点容器里的外点删掉，只保留内点进行后续的三角化或 PnP
         *  ) 
         * 为什么阈值是0.3 / 460 ? 我们希望误差小于 0.3 像素(由于是初始化，所以比较严格，如果是平时匹配可能会设置为1~3pix)， 
         * 而且因为输入是归一化坐标，所以阈值也要转成归一化单位， 归一化距离 ≈ 像素距离 / 焦距f
        */   
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        // 构造一个假的单位内参矩阵 (因为输入已经是归一化坐标了)
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        /**
         *  int cv::recoverPose (   从本质矩阵中恢复得到R、t，然后通过手性检查验证可能的姿态，校验三角化的3D点应该有正的深度。
         *                                  即该函数返回的R、t已经在内部去掉了另外三种错误的解，返回的是最合适的解。函数的返回值是通过了校验的内点个数
         *      InputArray  E,              本质矩阵
         *      InputArray  points1,        第一幅图像点的数组
         *      InputArray  points2,        第二幅图像点的数组
         *      InputArray  cameraMatrix,   相机内参矩阵K (如果输入点是归一化坐标，则传单位矩阵)
         *      OutputArray     R,          第一帧坐标系到第二帧坐标系的旋转矩阵
         *      OutputArray     t,          第一帧坐标系到第二帧坐标系的平移向量(尺度不可观，通常归一化为单位长度)
         *                                  注意: t 的长度没有意义（单目）。只有方向可用；尺度需要第三方信息（双目基线、IMU、已知尺度、轮速等）
         *      InputOutputArray    mask = noArray()  作为输入矩阵，可以传入 RANSAC 生成的内点 mask（只让 recoverPose 在这些点上工作），只有这些内点将被用于恢复位姿；
         *                                            作为输出矩阵，recoverPose 会进一步基于“正深度”等条件更新 mask， 它只包含通过了手性校验的内点。
         *  )  返回值: 通过了切利里性检查（Cheirality Check）的特征点数量。 
         *            判定标准：如果返回值太小，说明虽然算出了 R 和 t，但大部分点都在相机后面，这通常意味着计算失败或者特征点匹配质量极差。
        */
        // recoverPose函数内部做了哪些关键步骤？
        // 1. 对 本质矩阵E 进行 SVD 分解，得到 R1, R2, t 四组解 (R1,t), (R2,t), (R1,-t), (R2,-t)
        // 2. 对这四组解进行三角化，计算出3D点，然后进行手性检查（Cheirality Check），
        //    也就是检查三角化出来的3D点在两个相机坐标系下的深度是否都是正的，只有深度为正的3D点才是合理的
        // 3. 选择通过手性检查的3D点数量最多的那一组解，作为最终的 R 和 t 输出  
        // 4. 同时返回通过手性检查的内点数量 inlier_cnt，供调用者判断本次计算是否成功
        // 注意：recoverPose函数内部也使用了RANSAC，所以会进一步剔除一些外点，mask会被更新
        // ********************* //
        // 使用 recoverPose 什么时候会失败或不可靠?
        // 1. 纯旋转/平移很小（视差不足） E仍可能被估计出来 ，recoverPose 也能给出 R,t，但三角化不稳， t方向容易漂.初始化(尤其是单目尺度)通常失败
        // 2. 点分布退化    匹配点几乎共线、集中在小区域、或多数点在同一平面并且运动接近纯旋转，会造成不稳定
        // 3. 外点过多或阈值不合适  如果 RANSAC 阈值过严，mask 内点太少；过松则外点混进来，recoverPose 选错解或 inlier_cnt 虚高但不真实
        // 4. 坐标域不一致 输入是像素点却把 cameraMatrix=I; 输入是归一化点却又给真实 K; 这两种都会显著破坏 E/R/t 的正确性
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }
        // 将 OpenCV 格式转为 Eigen 格式，供 VINS 后续使用
        // OpenCV 通常使用的几何形式是：  P2 = R * P1 + t (1是参考帧，2是当前帧)
        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}



