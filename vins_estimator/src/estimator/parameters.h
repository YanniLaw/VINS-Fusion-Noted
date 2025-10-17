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
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0; // 焦距 为什么是固定的？
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;       // 初始化深度????
extern double MIN_PARALLAX;     // keyframe selection threshold (pixel) 关键帧之间的最小视差角（parallax）阈值
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

// 存放的camera到imu的外参 0 左目 1 右目
extern std::vector<Eigen::Matrix3d> RIC;    // R_imu_camera
extern std::vector<Eigen::Vector3d> TIC;    // t_imu_camera
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;                  // max solver itration time (ms), to guarantee real time
extern int NUM_ITERATIONS;                  // max solver itrations, to guarantee real time
extern std::string EX_CALIB_RESULT_PATH;    // 外参标定结果路径
extern std::string VINS_RESULT_PATH;        // vin 结果路径  OUTPUT_FOLDER + "/vio.csv" 
extern std::string OUTPUT_FOLDER;           // 结果输出的文件夹目录
extern std::string IMU_TOPIC;
extern double TD;               // initial value of time offset. unit: s. readed image clock + td = real image clock (IMU clock)
extern int ESTIMATE_TD;         // online estimate time offset between camera and imu
extern int ROLLING_SHUTTER;
extern int ROW, COL;            // 图像的尺寸
extern int NUM_OF_CAM;          // VIO系统中相机的数量( 单目:1  双目: 2 ）
extern int STEREO;              // 是否存在双目 NUM_OF_CAM == 2
extern int USE_IMU;
extern int MULTIPLE_THREAD; // 其实就是单独开启处理线程
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;  // 相机标定文件路径
extern int MAX_CNT;         // max feature number in feature tracking
extern int MIN_DIST;        // min distance between two features
extern double F_THRESHOLD;  // ransac threshold (pixel)
extern int SHOW_TRACK;      // publish tracking image as topic
extern int FLOW_BACK;       // perform forward and backward optical flow to improve feature tracking accuracy

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
