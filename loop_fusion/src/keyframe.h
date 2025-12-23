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

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;


class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();



	double time_stamp; 
	int index;										// 关键帧的索引ID(回环检测节点中)
	int local_index;							// 
	Eigen::Vector3d vio_T_w_i; 		// 该关键帧所在的VIO位置
	Eigen::Matrix3d vio_R_w_i; 		// 该关键帧所在的VIO姿态
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;					// 回环优化后该关键帧的世界位姿
	Eigen::Vector3d origin_vio_T;		// 回环优化前该关键帧的初始世界位姿
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;								// 该新关键帧对应的原始图像数据
	cv::Mat thumbnail;						// 该新关键帧对应的原始图像缩略图
	vector<cv::Point3f> point_3d; 			// 该新关键帧能够观测到的所有路标点在世界坐标系下的3d坐标
	vector<cv::Point2f> point_2d_uv;		// 该新关键帧能够观测到的路标点在当前帧上的有畸变的原始像素坐标
	vector<cv::Point2f> point_2d_norm;	// 该新关键帧能够观测到的路标点在当前帧相机坐标系下的去畸变后的相机归一化平面坐标
	vector<double> point_id;						// 该新关键帧能够观测到的路标点的VIO索引id
	vector<cv::KeyPoint> keypoints;							// 从该关键帧原始图片中新提取出的有畸变的特征关键点
	vector<cv::KeyPoint> keypoints_norm;				// 对应keypoints中的特征关键点在当前帧相机坐标系下的去畸变后的相机归一化平面坐标
	vector<cv::KeyPoint> window_keypoints;			// 该关键帧能够观测到的，已有的在滑动窗口中的路标点在当前帧上的有畸变的特征关键点，对应point_2d_uv
	vector<BRIEF::bitset> brief_descriptors;		// 对应keypoints中的特征关键点的描述子
	vector<BRIEF::bitset> window_brief_descriptors;		// 对应window_keypoints中的特征关键点的描述子
	bool has_fast_point;
	int sequence;

	bool has_loop;	// 是否已经检测到回环
	int loop_index;	// 回环索引
	Eigen::Matrix<double, 8, 1 > loop_info; // 该关键帧的IMU坐标系到闭环历史帧的IMU坐标系的相对位姿变换T^bi_bj，以及偏航角的偏移量。依次是(t_x,t_y,t_z,q_w,q_x,q_y,q_z,yaw)
};

