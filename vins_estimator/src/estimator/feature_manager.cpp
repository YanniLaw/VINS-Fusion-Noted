/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

// 设置两个相机的旋转外参R_i_c
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

//窗口中被多帧跟踪到过的特征的数量
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   选择关键帧的策略。只有关键帧才会被加入到滑窗内
 * @param[in]   frame_count：当前帧在滑动窗口中的id，从0开始
 * @param[in]   image：当前帧图像特征点的信息(feature_id, [camera_id, [x,y,z,u,v,vx,vy]])所构成的map，索引为feature_id
 * @param[in]   td：IMU和cam同步时间差
 * @return  bool true：边缘化最老帧；false：边缘化次新帧
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());  // image.size() 是当前帧图像中跟踪到的特征点数量
    ROS_DEBUG("num of feature: %d", getFeatureCount()); // 滑动窗口中的特征点数量(被多帧跟踪到过)
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;         // 统计当前帧图像中跟踪到的已经存在特征点列表中的特征点(旧点)数量
    last_average_parallax = 0;
    new_feature_num = 0;        // 统计当前帧图像数据中新增加的特征点数量(新点,此前没有在特征点列表中)
    long_track_num = 0;         // 统计当前帧中长时间跟随的特征点(旧点)数量
    // 遍历当前帧图像中每一个特征点的数据
    for (auto &id_pts : image) 
    {
        // 这里id_pts的结构是std::pair<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>> 即 feature_id - 特征点数据对数组
        // vector<pair<int, Eigen::Matrix<double, 7, 1>>> 是一个特征点在不同相机下的观测数据对(相机id, 特征数据)数组
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // id_pts.second[0].second其实就是特征点数据Eigen::Matrix<double, 7, 1>
        assert(id_pts.second[0].first == 0); // 左目的canera_id 为0 
        if(id_pts.second.size() == 2) // 双目模式
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
            assert(id_pts.second[1].first == 1); // 右目的canera_id 为1
        }
        // 根据feature_id判断在feature列表中是否已有该特征点
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature.end()) // 如果没有则说明是新点，新建一个FeaturePerId并添加到feature列表中
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); // 注意，这里传入的是frame_count，表示当前帧在滑动窗口中的id
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++; // 统计新点数量
        }
        else if (it->feature_id == feature_id) // 已经有了说明该特征点是旧点，直接添加该帧的观测数据
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;   // 统计旧点数量
            if( it-> feature_per_frame.size() >= 4) // 如果这个点已经被连续跟踪超过 4 帧，算作“稳定长特征”
                long_track_num++; // 统计长时间跟踪的旧点数量
        }
    }
    // 在以下情况下，应该把当前帧当作关键帧/需要更新: 
    // 1. 系统刚启动时，直接将前两帧图像作为关键帧
    // 2. 跟踪到的旧特征点数量较少(可能运动大、遮挡多、纹理差)
    // 3. 长时间跟踪到的旧特征点数量较少（稳定性不足）
    // 4. 新增点占比太高（意味着场景/视角变化大或跟踪在大量丢失）
    // 核心思想：当跟踪质量变差，或者画面发生剧变时，不要管视差够不够了，必须插入关键帧，否则系统会跟丢。
    // 如果当前图像帧跟踪匹配到的路标点的个数不足20，则将次新帧作为关键帧（关键帧是场景中具有代表性的帧）
    // 出现这种情况可能是当前帧移动较大或者存在遮挡，则次新帧要作为关键帧，否则可能会漏掉某个场景
    //if (frame_count < 2 || last_track_num < 20) // vins-mono 原始条件
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;
    // 当前帧跟踪到的旧特征点数量较多(跟踪良好)，则用另外的条件来进行判断
    for (auto &it_per_id : feature) // 遍历特征点列表中的每一个特征点数据FeaturePerId
    {
        // 该路标点的起始帧在是次新帧以前，且该路标点至少要被次次新帧和次新帧跟踪观测到过
        // (同时在 frame_count-2 和 frame_count-1 都出现的特征（即连续两帧都跟踪到了这个特征点，因为后面要去计算他们的视差）)
        // 平均视差要在“连续跟踪成功的同一批点”上统计才有意义
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // 计算该路标点在次次新帧和次新帧的观测数据的视差量(上一帧和上上帧之间的视差)
            parallax_num++;
        }
    }
    // 没有任何点满足“连续两帧都有观测”的条件，就没法算可靠平均视差，此时返回 true 相当于“保守策略：把它当作关键帧/触发更新”
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH; // 像素为单位的平均视差, 乘 FOCAL_LENGTH（像素单位焦距）后变成大致“像素级视差”
        // 视差足够大 → 基线足够 → 三角化/约束更强 → 值得把当前帧作为关键帧（或至少触发一次优化）
        // 视差太小 → 相机几乎没平移（或主要是旋转）→ 加入新关键帧价值有限，可能跳过或延迟
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

// 在 FeatureManager 当前维护的所有特征点中，筛出那些同时在两帧 frame_count_l 与 frame_count_r 都有观测的特征，并返回它们在两帧的归一化平面坐标
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) // 该特征点在两帧中都有观测
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame; // 索引换算
            int idx_r = frame_count_r - it.start_frame;
            // feature_per_frame 是一个局部 vector，它存的第一帧对应的是系统的 start_frame
            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;
}

/**
 * @brief 获取滑动窗口中被多帧(>=4)跟踪到过的特征的数量，并将这些特征的深度值以向量形式返回（注意是逆深度）
 *  这里为什么用逆深度？
 * 1. 压缩状态变量维度（降低计算量）: 直接优化每个特征点的3D位置需要为每个点维护三个参数（x, y, z），而优化深度的倒数只需要一个参数。
 *    这在滑动窗口中可以显著减少状态变量的维度，降低优化的计算复杂度。
 * 2. 数值稳定性更好，尤其对远点和小视差点：直接优化深度值可能会导致数值不稳定（尤其是当深度较大时），
 *    而优化深度的倒数可以使问题更稳定，且在某些情况下更适合优化器的收敛特性。
 *    普通深度z对远点会变得很大，线性化时很容易病态；而逆深度λ=1/z对远点会自然接近0，甚至“接近无穷远”的点也还能用一个接近0的小量来表示。
 *    对低视差、远距离特征的条件数更友好，优化器更容易处理。
 * 3. 和首观测帧锚定的重投影模型更匹配: VINS 不是把地图点当成独立对象满世界漂，而是把特征“挂”在它的首观测帧上，再通过后续帧的重投影来约束这个标量.
 * 4. 更适合滑窗中的 Schur 消元和边缘化: VINS-Fusion 的窗口里，最老帧会不断被边缘化。若特征是用全局 3D 点参数化，很多点会和大量历史位姿强耦合;
 *    而用 host-frame inverse depth，特征和“首观测帧 + 后续少数观测帧”之间的关系更局部，做 Schur 补和先验传递更方便
 * @return VectorXd 
 */
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // 滑动窗口中被多帧(>=4)跟踪到过的特征点的数量
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth; // 逆深度
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 参考 GlobalSFM::triangulatePoint，输入两帧的投影矩阵和两帧的特征点观测，输出三维点坐标
void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

/**
 * @brief 利用“已有深度的老特征”构造 3D–2D 对应，用 PnP(Perspective-n-Point) 算法求解当前帧的初始位姿
 * 当系统已经有了一些带深度的 3D 特征点（比如通过双目第一帧三角化出来的点），新来了一帧图像时，
 * 我们就可以利用这些已知 3D 点在当前帧的 2D 投影位置，反推算出当前帧相机的空间位姿
 * 只有在双目模式下才会调用该函数
 * 总而言之，就是利用当前帧跟踪到的“老特征点”来进行PNP求解当前帧的位姿(初始解，后面会进行优化的)
 * @param frameCnt 当前帧在滑窗中的id
 * @param Ps 
 * @param Rs 
 * @param tic 外参
 * @param ric 
 */
void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    // 需要第一帧数据在初始化时通过双目三角化得到路标点的深度，
    // 所以第二帧才能使用 PnP 算法来求解当前帧的位姿
    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &it_per_id : feature)
        {
            if (it_per_id.estimated_depth > 0) // 已经三角化计算出了深度信息
            {
                // 该特征在当前帧 frameCnt 里还能看到
                // 如果这条特征从 start_frame 开始被连续跟踪，那么它在当前帧里的观测应该是 feature_per_frame[index]
                // 如果没有当前帧的 2D 观测，这个点就无法用来构建 3D-2D 的 PnP 匹配对，直接被过滤掉
                int index = frameCnt - it_per_id.start_frame;
                if((int)it_per_id.feature_per_frame.size() >= index + 1) // 不符合这个条件说明该特征点在当前帧没有观测数据，无法用于 PnP 求解
                {
                    // 先得到该点在首观测左相机坐标系中的三维坐标，再通过外参变换到首帧观测imu坐标系
                    // feature_per_frame[0] 是该特征点在首观测帧的观测数据，point是归一化平面坐标，estimated_depth是深度
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0];
                    // 再通过首观测帧的 IMU 位姿变到世界系
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    // 当前帧左目的2d观测
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d); 
                }
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam  用上一帧的位姿作为初始值
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D)) // PNP求解的是 当前左目在世界系中的位姿
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose(); 
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}

/**
 * @brief 对还没有深度信息的路标点进行三角化求深度（SVD分解）
 * 其实是一个灵活的深度计算策略——双目靠空间基线，单目靠时间基线，观测多时上 SVD 求最优解
 * 在VINS-Fusion 的后端里，特征是按“首观测帧的逆深度”进入滑窗状态的，所以这里不是在求最终优化结果，而是在给后续非线性优化提供一个初始depth seed
 * @param frameCnt 当前帧在滑窗中的id
 * @param Ps 滑窗中每帧的平移向量（带尺度的平移） Ps[i] 是第 i 帧在世界坐标系下的平移向量，i 从 0 到 frameCnt
 * @param Rs 滑窗中每帧的旋转矩阵 Rs[i] 是第 i 帧在世界坐标系下的旋转矩阵，i 从 0 到 frameCnt
 * @param tic 相机系到imu系的平移向量
 * @param ric 相机系到imu系的旋转矩阵
 */
void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历滑窗中的每个路标点，求出该路标点在它start_frame帧坐标系下的深度。
    // 在初始化恢复出尺度s之后，Ps[]是带有尺度的平移，因此这里求出的深度是真实的物理深度
    for (auto &it_per_id : feature)
    {
        // 单目模式下，如果是刚初始化，那么此时会清空掉所有的估计深度值
        if (it_per_id.estimated_depth > 0) // 该路标点的深度值如果大于0，说明该点已被三角化过了
            continue;
        // 双目模式下，同一时刻直接三角化(因为双目同一时刻两帧图像之间的基线是固定的，三角化结果更稳定)
        if(STEREO && it_per_id.feature_per_frame[0].is_stereo)
        {
            int imu_i = it_per_id.start_frame; // 直接用首帧观测进行三角化
            Eigen::Matrix<double, 3, 4> leftPose; // 左目,这里的Pose是投影矩阵，即T_c_w
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // t_w_c = t_w_i + R_w_i * t_i_c
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];  // R_w_c = R_w_i * R_i_c
            leftPose.leftCols<3>() = R0.transpose(); // 下面是求T_c_w
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            //cout << "left pose " << leftPose << endl;

            Eigen::Matrix<double, 3, 4> rightPose; // 右目，同一时刻两帧图像之间的基线是固定的，三角化结果更稳定
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;
            //cout << "right pose " << rightPose << endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2); // 因为是归一化的坐标
            point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;
            // point3d 是三角化得到的路标点在世界坐标系下的坐标
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            // P_c = T_c_w * P_w = [R_c_w | t_c_w] * P_w = R_c_w * P_w + t_c_w
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>(); // 把三角化得到的世界点重新变回左相机坐标系
            double depth = localPoint.z(); // depth就是该路标点在首观测帧的左目坐标系下的z值，也就是深度
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        else if(it_per_id.feature_per_frame.size() > 1) // 单目模式下，该路标点至少被两帧观测到过，用前后两帧进行快速三角化
        {
            // 用该特征的前两次观测去做三角化
            // 它只用前两次观测，没用整条轨迹。所以这不是最优多视图三角化，而是一个“够快够用”的初始值生成器
            // 它默认第二个观测就是 imu_i + 1 这帧
            int imu_i = it_per_id.start_frame; // 首帧观测
            Eigen::Matrix<double, 3, 4> leftPose; // 这里是投影矩阵，即T_c_w
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];    // t_w_c = t_w_i + R_w_i * t_i_c
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];                // R_w_c = R_w_i * R_i_c
            leftPose.leftCols<3>() = R0.transpose();                // R_c_w = R_w_c^T
            leftPose.rightCols<1>() = -R0.transpose() * t0;         // t_c_w = -R_w_c^T * t_w_c
            // 第二帧观测，默认就是首帧观测的下一帧
            imu_i++;
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);  // 第一帧观测的平面归一化坐标
            point1 = it_per_id.feature_per_frame[1].point.head(2);  // 第二帧观测的平面归一化坐标
            // point3d 是三角化得到的路标点在世界坐标系下的坐标
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>(); // 将三角化结果变到首帧观测的左相机坐标系
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        // 下面的内容似乎走不到哦  这里是原来VINS MONO的代码
        // 多视角svd三角化，该路标点至少被四帧观测到过，才能进行三角化
        // 当有多次观测时，直接使用简单的2帧三角化可能精度不够，系统会使用直接线性变换（DLT）结合 SVD 分解来求取最优解
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();
        // 每个视角贡献两条独立方程，把所有观测堆起来后，对 A 做 SVD，取最小奇异值对应的右奇异向量：
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            feature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

// 边缘化掉最老帧时，处理特征点被观测的帧号，并将起始帧是最老帧的特征点的深度值转移到原先第二老的帧的相机坐标系下
// marg_R、marg_P是滑动窗口中原先最老的帧（被边缘化掉的帧）的位姿(c->w)
// new_R、new_P是滑动窗口中原先第二老的帧（现在是最老的帧）的位姿(c->w)
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // 对feature容器进行更新，因为最老帧被marg掉了
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0) // 观测到该特征点的起始帧不是最老帧（即0帧），则将起始帧号减一，因为在下面代码中会移除0帧
            it->start_frame--;
        else // 观测到该特征点的起始帧是被边缘化的最老帧
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point; // 该特征点在最老帧（被边缘化的帧）相机坐标系下的去畸变后的相机归一化平面坐标 
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 移除边缘化的最老帧的信息
            if (it->feature_per_frame.size() < 2) // 如果该特征点只被最老帧和第二老帧观测到，则移除该特征
            {
                feature.erase(it);
                continue;
            }
            else
            {   // estimated_depth 本身就是位于起始帧下的深度值
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth; // 该特征点在最老帧（被边缘化的帧）相机坐标系下的的三维坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;  // 该特征点在世界坐标系下的三维坐标
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // 将其转换到第二老的帧(滑窗后的第0帧)相机坐标系下的三维坐标
                double dep_j = pts_j(2); // 深度值
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else // 无效的深度值，重置为初始值
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

// 边缘化最老帧时，直接将路标点所保存的帧号向前滑动（窗口中处理的是相邻的帧）
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0) // 观测到该路标点的起始帧不是最老帧（滑窗中的0帧），则将起始帧号减一，因为在下面代码中会移除0帧
            it->start_frame--;
        else  // 观测到该路标点的起始帧是最老帧，则直接删除该路标点在最老帧的观测数据，
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 移除0帧的信息
            if (it->feature_per_frame.size() == 0)
                feature.erase(it); // 如果移除了之后为空，则删除该路标点
        }
    }
}

// 边缘化次新帧时，对路标点在次新帧的信息进行移除处理
// 传入的frame_count其实就是WINDOW_SIZE
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count) // 观测到该路标点的起始帧是当前最新帧（frame_count帧），则将起始帧号减一，这样就将其滑动成为了次新帧
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1) // 如果该路标点在次新帧之前就已经跟踪结束了，则什么都不做
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j); // 如果该路标点在次新帧中仍被跟踪观测到了，则移除次新帧的信息
            if (it->feature_per_frame.size() == 0)
                feature.erase(it); // 如果移除次新帧信息之后图像观测为空，则删除该路标点
        }
    }
}

// 计算某个路标点it_per_id，在次次新帧和次新帧的观测数据的视差量
// frame_count为当前帧在滑动窗口中的id
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // 次次新帧(上上帧)
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // 次新帧(上一帧)

    double ans = 0;
    Vector3d p_j = frame_j.point; // 该路标点在该帧相机坐标系下的去畸变后的相机归一化平面坐标

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;
    // 旋转平移补偿
    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}