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

#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

// 已知两个视角的相机投影矩阵（Pose0, Pose1，其实就是T_c_w）和对应的归一化平面坐标（point0, point1），求解出该特征点在空间中的3D坐标
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}
/**
 * @brief 给定两个已知位姿的帧（Frame0 和 Frame1），遍历滑窗中所有的特征点，找出同时被这两帧观测到、且尚未计算出3D坐标的点，对其进行三角化
 * 
 * @param frame0 Frame0在滑窗中的索引
 * @param Pose0  Frame0的坐标变换	T_c0_w
 * @param frame1 Frame1在滑窗中的索引
 * @param Pose1  Frame1的坐标变换	T_c1_w
 * @param sfm_f 滑窗中的所有特征点
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++) // 遍历滑窗中的所有特征点
	{
		if (sfm_f[j].state == true) // 如果这个特征点之前已经算出来 3D 坐标了（state == true），就跳过
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) // 遍历该特征点的所有观测记录
		{
			if (sfm_f[j].observation[k].first == frame0) 	// 检查是否在 frame0 中出现
			{
				point0 = sfm_f[j].observation[k].second;		// 获取去畸变归一化平面坐标
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)	// 检查是否在 frame1 中出现
			{
				point1 = sfm_f[j].observation[k].second;		// 获取去畸变归一化平面坐标
				has_1 = true;
			}
		}
		// 这里只要两帧都有观测就三角化
		// 不做额外质量控制: 1. 不检查三角化后的点是否在两相机前方（cheirality / 正深度）; 2. 不检查视差是否足够、夹角是否太小; 3. 不检查重投影误差
		// 局限: 容易生成"坏点"， 当两帧基线太小、或 Pose 估计误差大、或匹配点有外点时，会产生深度为负/深度非常大/重投影误差很大，这些坏点如果不被后续 BA/鲁棒核压住，会拖累初始化稳定性
		// 如果你在初始化阶段经常 SFM/BA 不稳定，通常是三角化点质量问题， 调试时尽量加三项检查: 
		// 1. 正深度检查		 计算z0,z1是否都大于0(在相机坐标系下)，否则不置state = true
		// 2. 重投影误差门限 	把三角化点投回两帧，看误差是否 < 某阈值（归一化域阈值可换算像素）， 超过则拒绝该点
		// 3. 最小视差/夹角 	如果两帧观测差很小（或射线夹角过小），三角化条件数差，直接跳过
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2); // note: 这里求出的是以滑窗中第l帧为世界坐标系的三维点坐标
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/**
 * @brief   纯视觉SFM，求解滑动窗口中的每一图像帧的相对于l帧的位姿和路标点坐标
 * 在已知两帧（参考帧 l 和最新帧 frame_num-1）相对位姿的情况下，利用增量式 SfM（Structure from Motion）的方法，
 * 恢复滑动窗口内所有帧的位姿以及所有特征点的 3D 坐标，并进行全局 BA 优化
 * @param[in]   frame_num：滑动窗口内的图片总帧数（frame_count + 1），所以当前帧在滑窗中的索引是frame_num -1 
 * @param[out]  q：滑动窗口内图像帧的旋转四元数q（相对于l帧）
 * @param[out]	T：滑动窗口内图像帧的平移向量T（相对于l帧）
 * @param[in]  	l：l帧
 * @param[in]  	relative_R：当前帧到l帧的旋转矩阵
 * @param[in]  	relative_T：当前帧到l帧的平移向量
 * @param[in]  	sfm_f：当前滑动窗口中的所有特征点
 * @param[out]  sfm_tracked_points：在sfm中所有成功三角化后的路标点的id和三维坐标
 * @return  bool true:sfm求解成功
*/
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size(); // 当前滑窗中特征点的数量
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	// 设定滑窗中第l帧为世界坐标系原点(非常重要的假设)，根据当前帧到l帧的relative_R，relative_T，得到当前帧位姿
	// 即相机系到世界系的坐标变换
	q[l].w() = 1; // T_w_c = I
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R); // 当前帧T_w_current = T_l_current
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame，世界系相对相机系的坐标变换
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4]; // 数组形式，ceres 优化的目标
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		// 然后从滑窗中第l+1帧开始利用已成功三角化的3D点寻找3D-2D的匹配从而通过PNP求解出位姿
		// 所以在初始化时，相机移动不能过快，否则可能没有重叠区域，就会没有被共同观测到的路标点，这样也不能完成solveFrameByPnP，从而求不出l+1帧位姿
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1]; // 使用上一帧 (i-1) 的位姿作为初始猜测(连续性假设)
			Vector3d P_initial = c_Translation[i - 1];
			// 利用已经三角化出来的3D点和当前帧对应的2D观测，求解该帧的Pose
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		// 循环先进入到这里，对滑窗中第l帧与当前帧三角化生成一部分3D点
		// 每算出一个 Pose[i]，就立刻用它和 当前帧 做两帧三角化，把能三角化的点尽量 triangulate 出来，为后续 PnP 提供更多 3D点
		// 为什么总是和当前帧做三角化？ 因为 当前帧 和较早帧通常有更大基线，三角化更稳，能更快得到“够用的 3D 点集”支撑 PnP 链式求解
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2 补充三角化,让其他帧和滑窗中第l帧(参考帧)也进行三角化
	// 作用：提高点覆盖率。某些点可能不在 当前帧(最新帧) 中出现，或者 (i, frame_num - 1) 几何不佳，这里用 (l,i) 再补一次。
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	// 利用滑窗中第l帧(参考帧)及其右边已经建立好的丰富3D结构，反向求解左边帧的位姿，并进行三角化
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	// 至此得到了滑动窗口中所有图像帧的位姿(其实是T_c_w,即Pose)，以及绝大部分特征点的3D坐标
	//5: triangulate all other points
	// 兜底: 继续三角化其它未恢复的3d路标点。注意，这里恢复出来的路标点坐标都是基于滑窗中第l帧坐标系(参考帧)
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2) // 至少被两帧观测到才能三角化
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);	// 用“首末观测帧”两帧三角化， 保证两图像帧之间有较大的平移
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA 使用ceres对滑动窗口中的所有图像帧的位姿以及路标点进行BA优化
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	// 添加参数块
	for (int i = 0; i < frame_num; i++) // 滑动窗口中的所有帧。这里的位姿是世界系相对相机系的变换(滑窗中第l帧到其它每一帧的变换)
	{
		//double array for ceres, ceres中必须使用数组存储参数
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization); // 注意，c_translation和c_rotation都是二维数组，因此传给ceres函数的是指针
		problem.AddParameterBlock(c_translation[i], 3);
		// 固定帧 l 的 R 和 t：确定世界坐标系原点
		// 固定帧 End(当前帧)的 t：确定尺度 (Scale)。这是单目 SfM 能够收敛的关键
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]); // 以滑窗中第l帧做为参考世界坐标系，所以固定l帧的位姿，不做优化
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]); // 固定参考帧和当前帧的平移，因为上面的所有位姿估计都依赖于这两个帧
			// 核心原因: 单目BA的尺度不可观，这里尺度就以初始化时 relative_T 的长度为准（仍是任意尺度，但固定成一致的尺度），后面再由IMU对齐真实尺度。
		}
	}
	// 添加重投影误差残差块
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true) // 只对三角化成功的点进行BA优化
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first; // 该特征点被观测到的帧在滑窗中的索引
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y()); // 该特征点在该帧的归一化平面坐标
				// 添加残差项。该残差项对应的待优化变量块有：世界系(滑窗中第l帧)到图像系的位姿变换、路标点在世界系(滑窗中第l帧)中的3d坐标
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], // 这个变量l是局部变量，不是滑窗中参考帧l
    								sfm_f[i].position);	 
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2; // 初始化阶段BA求解时间不宜过长
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	// 把 BA 结果转回 q[i], T[i]（回到 camera→world）
	// 将优化后的相机位姿结果放在q[]、T[]数组，3d坐标保存在sfm_tracked_points
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; // 这里还是世界系相对相机系的变换
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse(); // 现在相机系相对世界系的变换，即各图像帧到l帧坐标系的变换 // R_wc = R_cw^T
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		// t_wc = -R_wc * t_cw
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state) // 只返回那些成功三角化的路标点
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

