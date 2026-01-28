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

#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
    bool state; // 路标点的状态（是否被成功三角化了）
    int id;			// 路标点的id(唯一)
    vector<pair<int,Vector2d>> observation; // 观测到该路标点的所有图像帧在滑动窗口中的id以及该路标点在该帧相机坐标系下的去畸变后的相机归一化平面坐标
    double position[3];	// 成功三角化之后的3d坐标。注意，这个3d坐标的参考坐标系是l帧相机坐标系
    double depth;				// 深度
};

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	// 使用仿函数来进行误差的计算。residuals是计算出来的残差变量
	// camera_R、camera_T、point是待优化变量块，分别对应了在调用函数AddResidualBlock()时传入的参数c_rotation[obs_id]、c_translation[obs_id]、sfm_f[i].position。
	// 并且这里待优化变量块的传入方式应和AddResidualBlock()一致，即若在AddResidualBlock()中是一次性传入变量块数组指针，
	// 此处也应该一次性传入变量块数组指针；若变量块是依次传入的，此处也应该依次传入。且要保证变量块的传入顺序一致。
	// 该函数的作用：
	// 将世界坐标系下（初始化时是第l帧）的3d路标点坐标point变换到当前相机帧下，即预测得到了一个相机归一化坐标；再与观测量observation做差值，得到重投影误差
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2]; // 坐标变换R*p+t。将世界系（l帧）下的路标点变换到当前帧坐标系下
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2]; // 转换到相机归一化平面坐标系
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v); // 计算重投影误差
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
		// 自动求导。模板类AutoDiffCostFunction的构造函数原型如下：
		// AutoDiffCostFunction<CostFunctor, int residualDim, int paramDim>(CostFunctor* functor);
		// 模板参数依次是：仿函数类型CostFunctor，残差的维数residualDim、每个待优化变量块的维数paramDim；
		// 函数参数类型是：仿函数指针CostFunctor*。
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>( // 这里的几个数值参数指的是2维的残差，4维的相机旋转四元数，3维的平移，3维的路标点
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v; // 观测量：路标点在该帧相机归一化平面坐标系下的坐标
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};